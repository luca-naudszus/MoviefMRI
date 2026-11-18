#!/usr/bin/env python3
"""
fmriprep_sub.py is a wrapper around fmriprep that queries the bids directory for
new participants and then runs participant-level fmriprep jobs on the compute
cluster (one job per subject; all sessions of a subject are processed together,
which is how fmriprep builds its subject-level anatomical template).

The fMRIPrep image, FreeSurfer license and TemplateFlow directory are read from
versions.sh (by default the one next to this script), so there is nothing to
source before running it. Options given on the command line take precedence.

When all jobs have finished, every subject has its own <outputdir>/sub-XX.html
report and preprocessed data in <outputdir>/sub-XX/.
"""

import os
import shutil
import subprocess
import argparse
import textwrap
import shlex
from pathlib import Path


def is_done(bidsdir: Path, outputdir: Path, sub_id: str) -> tuple[bool, int, int, list]:
    """A subject is considered done if the html-report exists and there is a confounds-file for every func/*_bold run"""

    nbold     = len(list((bidsdir/sub_id).glob('**/func/*_bold.nii*')))
    confounds = len(list((outputdir/sub_id).glob('**/func/*_desc-confounds_timeseries.tsv')))
    crashes   = list((outputdir/sub_id/'log').glob('**/crash-*'))
    report    = (outputdir/f"{sub_id}.html").is_file()

    return report and confounds >= nbold, confounds, nbold, crashes


def read_versions(path: Path) -> dict:
    """Let bash evaluate versions.sh (so that $HOME, ${...} etc. are expanded) and return the resulting variables"""

    if not path.is_file():
        print(f"ERROR: versions file not found: {path}")
        exit(1)
    process = subprocess.run(['bash', '-c', 'set -a; source "$1" >/dev/null; env -0', '_', str(path)], capture_output=True, text=True)
    if process.returncode != 0:
        print(f"ERROR: Could not read {path}:\n{process.stderr}")
        exit(1)
    return dict(item.split('=', 1) for item in process.stdout.split('\0') if '=' in item)


def check_file(path: str, name: str, envvar: str, is_dir: bool = False) -> Path:
    """Make sure that a required file / directory exists"""

    if not path:
        print(f"ERROR: No {name} given. Define {envvar} in versions.sh, or pass the path explicitly")
        exit(1)
    path = Path(path).expanduser()
    if not (path.is_dir() if is_dir else path.is_file()):
        print(f"ERROR: {name} not found: {path}")
        exit(1)
    return path.resolve()


def main(bidsdir: str, outputdir: str, image: str, workroot: str, license: str, templateflow: str,
         participants=(), force=False, mem_gb=32, nthreads=8, walltime=24, args='', qargs='', dryrun=False, nosub=False, skip=True):

    bidsdir   = Path(bidsdir).resolve()
    outputdir = Path(outputdir).resolve() if outputdir else bidsdir/'derivatives/fmriprep'

    # Everything below comes from versions.sh (unless given explicitly)
    image   = check_file(image,   'fMRIPrep image',          'FMRIPREP_IMAGE')
    license = check_file(license, 'FreeSurfer license file', 'FS_LICENSE')
    if templateflow:
        templateflow = check_file(templateflow, 'TemplateFlow directory', 'TEMPLATEFLOW_HOME', is_dir=True)
    print(f"Image:        {image}\nLicense:      {license}\nTemplateFlow: {templateflow or '(none: templates are downloaded by the jobs)'}")

    # Map the bids subject-directories
    if not participants:
        subjects = sorted(p for p in bidsdir.glob('sub-*') if p.is_dir())
    else:
        subjects = [bidsdir/(p if p.startswith('sub-') else f"sub-{p}") for p in participants]

    # Get the names of the already running / scheduled jobs (once)
    running = subprocess.run('squeue -u $USER -h -o %j', shell=True, capture_output=True, text=True).stdout.split()

    # Loop over bids subject-directories and submit a job for every (new) subject
    for n, subject in enumerate(subjects):

        if not subject.is_dir():
            print(f">>> Directory does not exist: {subject}")
            continue

        # Skip subjects with data that is not (yet) downloaded (e.g. broken datalad/git-annex symlinks)
        niifiles = list(subject.glob('**/*.nii*'))
        missing  = [f for f in niifiles if not f.exists()]
        if missing:
            print(f">>> Skipping {subject}: {len(missing)} file(s) not downloaded")
            continue

        sub_id  = subject.name
        jobname = f"fmriprep_{sub_id}"

        # Never touch subjects whose job is still running / scheduled (also not with --clobber)
        if skip and jobname in running:
            print(f"\n--> Skipping already running / scheduled job ({n+1}/{len(subjects)}): {jobname}")
            continue

        if workroot:
            workdir = Path(workroot)/sub_id
        else:
            scratch = os.getenv('SCRATCH') or f"/cluster/scratch/{os.getenv('USER')}"
            workdir = Path(scratch)/'fmriprep_work'/sub_id

        done, nconfounds, nbold, crashes = is_done(bidsdir, outputdir, sub_id)
        print(f"\n>>> Found {nconfounds}/{nbold} preprocessed bold runs for: {sub_id}")
        if crashes:
            print(f"    NB: {len(crashes)} crash-file(s) found in: {outputdir/sub_id/'log'}")
        if nbold == 0:
            print(f"--> No func/*_bold data found, skipping job ({n+1}/{len(subjects)}): {sub_id}")
            continue

        if done and not force:
            print(f"--> Nothing to do for job ({n+1}/{len(subjects)}): {sub_id}")
            continue

        # Start with a clean slate if we are forcing to reprocess the data (as presumably something went wrong or has changed)
        if force:
            for target in (workdir, outputdir/sub_id):
                if target.is_dir():
                    print(f"Cleaning: {target}")
                    if not dryrun:
                        shutil.rmtree(target, ignore_errors=True)
            report = outputdir/f"{sub_id}.html"
            if report.is_file():
                print(f"Cleaning: {report}")
                if not dryrun:
                    report.unlink()

        # Bind the license (and optionally a pre-fetched TemplateFlow directory) into the container
        binds = f"--bind {workdir}/tmp:/tmp --bind {license}:/opt/freesurfer/license.txt"
        envs  = ''
        if templateflow:
            binds += f" --bind {templateflow}:/templateflow"
            envs   = '--env TEMPLATEFLOW_HOME=/templateflow'

        # Leave some memory headroom for fmriprep, so that slurm does not kill the job
        mem_mb = int(mem_gb * 1024 * 0.9)

        # Generate the submit-command (Euler only accepts memory per CPU)
        mem_per_cpu = -(-mem_gb * 1024 // nthreads)
        submit = f"sbatch --job-name={jobname} --cpus-per-task={nthreads} --mem-per-cpu={mem_per_cpu}M --time={walltime}:00:00 {qargs}"

        # Generate the fmriprep-job. All paths are written into the job literally, so it does not need to source anything
        # and the job script itself documents exactly which image and settings were used
        job = textwrap.dedent(f"""\
                    #!/bin/bash
                    ulimit -v unlimited
                    cd {Path.cwd()}
                    mkdir -p {workdir}/tmp
                    apptainer exec --cleanenv {binds} {envs} {image} fmriprep {bidsdir} {outputdir} participant -w {workdir} --participant-label {sub_id[4:]} --fs-license-file /opt/freesurfer/license.txt --nprocs {nthreads} --omp-nthreads {nthreads} --mem_mb {mem_mb} --notrack {args}""")

        # Submit the job to the compute cluster ('EOF' is quoted so that $variables are expanded inside the job, not at submission)
        if nosub:
            workdir.mkdir(parents=True, exist_ok=True)
            command = job
        else:
            command = f"{submit} <<'EOF'\n{job}\nEOF"
        print(f"--> {'Running' if nosub else 'Submitting'} job ({n+1}/{len(subjects)}):\n{command}")
        if not dryrun:
            process = subprocess.run(command, shell=True, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"ERROR {process.returncode}: Job {'run' if nosub else 'submission'} failed\n{process.stderr}\n{process.stdout}")
            elif not nosub:
                print(process.stdout.strip())

    if not subjects:
        print(f"No BIDS subject folders found in {bidsdir}")
    elif dryrun:
        print('\n----------------\nDone! NB: The printed jobs were not actually submitted')
    elif nosub:
        print('\n----------------\nDone!')
    else:
        print('\n----------------\n'
              f"Done! Now wait for the jobs to finish... Check that e.g. with this command:\n\n  squeue -u {os.getenv('USER')} -o %j | grep fmriprep_sub\n\n"
              f"The subject reports will be in: {outputdir}/sub-*.html\n"
              'Rerun this script afterwards to see which subjects still need (re)processing\n')


# Shell usage
if __name__ == "__main__":

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter, description=textwrap.dedent(__doc__),
                                     epilog='Any unrecognized options are forwarded directly to fmriprep\n\n'
                                            'for more information see:\n'
                                            '  fmriprep -h\n'
                                            '  https://fmriprep.org/en/stable/usage.html\n\n'
                                            'examples:\n'
                                            '  fmriprep_sub.py /cluster/work/gess/sbs/MoviefMRI/data\n'
                                            '  fmriprep_sub.py /cluster/work/gess/sbs/MoviefMRI/data -p 01 02 --output-spaces MNI152NLin2009cAsym:res-2 T1w\n'
                                            '  fmriprep_sub.py /cluster/work/gess/sbs/MoviefMRI/data --fs-no-reconall --skip-bids-validation\n'
                                            '  fmriprep_sub.py /cluster/work/gess/sbs/MoviefMRI/data -c -p sub-03 -m 48 -n 12 -t 36\n'
                                            '  fmriprep_sub.py /cluster/work/gess/sbs/MoviefMRI/data -d\n\n'
                                            'Author:\n'
                                            '  Luca A. Naudszus (adapted from mriqc_sub by Marcel Zwiers)\n')
    parser.add_argument('bidsdir',              help='The bids-directory with the subject data')
    parser.add_argument('-o','--outputdir',     help='The fmriprep output-directory (default = bidsdir/derivatives/fmriprep)', default='')
    parser.add_argument('--versions',           help='The file that pins the tool versions and shared paths', default=str(Path(__file__).resolve().parent/'versions.sh'))
    parser.add_argument('-i','--image',         help='Path to the fMRIPrep apptainer image (default = FMRIPREP_IMAGE from versions.sh)', default='')
    parser.add_argument('-l','--license',       help='Path to the FreeSurfer license.txt file (default = FS_LICENSE from versions.sh)', default='')
    parser.add_argument('--templateflow',       help='Path to a pre-fetched TemplateFlow directory (default = TEMPLATEFLOW_HOME from versions.sh)', default='')
    parser.add_argument('-w','--workdir',       help='The root working-directory where intermediate files are stored (default = $SCRATCH/fmriprep_work)', default='')
    parser.add_argument('-p','--participants',  help='Space separated list of selected sub-# names / labels to be processed. Otherwise all subjects in the bidsfolder will be selected', nargs='+')
    parser.add_argument('-c','--clobber',       help='If this flag is given subjects will be processed with a clean working directory and their existing fmriprep output removed. Otherwise finished subjects will be skipped. Running / scheduled jobs are never touched', action='store_true')
    parser.add_argument('-r','--resubmit',      help='If this flag is given then jobs are submitted, regardless of already running or scheduled jobs with the same name, otherwise job submission is skipped', action='store_false')
    parser.add_argument('-m','--mem_gb',        help='Required amount of memory in GB (total per job)', default=32, type=int)
    parser.add_argument('-n','--nthreads',      help='Number of CPUs per job', default=8, type=int)
    parser.add_argument('-t','--time',          help='Required walltime in hours', default=24, type=int)
    parser.add_argument('-q','--qargs',         help='Additional (opaque) arguments that are passed to the resource manager (NB: Use quotes and include at least one space character to prevent overearly parsing)', type=str, default='')
    parser.add_argument('--nosub',              help='Add this flag to run the fmriprep commands locally, without submitting them (useful for debugging)', action='store_true')
    parser.add_argument('-d','--dryrun',        help='Add this flag to just print the fmriprep sbatch commands without actually submitting them (useful for debugging)', action='store_true')

    # Parse only what we know
    args, passthrough = parser.parse_known_args()

    # Catch the case where 'python' is the Python inside a container (e.g. after sourcing environment.sh): sbatch is not available there
    if os.getenv('APPTAINER_CONTAINER') or os.getenv('SINGULARITY_CONTAINER'):
        print('ERROR: This Python is running inside a container, where sbatch is not available (did you source environment.sh?).\n'
              'Run this script from a fresh shell instead; it reads versions.sh by itself')
        exit(1)

    versions = read_versions(Path(args.versions))

    main(bidsdir      = args.bidsdir,
         outputdir    = args.outputdir,
         image        = args.image        or versions.get('FMRIPREP_IMAGE', ''),
         workroot     = args.workdir,
         license      = args.license      or versions.get('FS_LICENSE', ''),
         templateflow = args.templateflow or versions.get('TEMPLATEFLOW_HOME', ''),
         participants = args.participants,
         force        = args.clobber,
         mem_gb       = args.mem_gb,
         nthreads     = args.nthreads,
         walltime     = args.time,
         args         = " ".join(shlex.quote(a) for a in passthrough),
         qargs        = args.qargs,
         dryrun       = args.dryrun,
         nosub        = args.nosub,
         skip         = args.resubmit)