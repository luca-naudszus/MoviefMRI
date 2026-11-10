#!/usr/bin/env python3
"""
fmriprep_sub.py is a wrapper around fmriprep that queries the bids directory for
new participants and then runs participant-level fmriprep jobs on the compute
cluster (one job per subject; all sessions of a subject are processed together,
which is how fmriprep builds its subject-level anatomical template).

When all jobs have finished, every subject
has its own <outputdir>/sub-XX.html report and preprocessed data in
<outputdir>/sub-XX/.
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


def main(bidsdir: str, outputdir: str, image: str, envscript: str, workroot: str, license: str, templateflow: str,
         participants=(), force=False, mem_gb=32, nthreads=8, walltime=24, args='', qargs='', dryrun=False, nosub=False, skip=True):

    bidsdir   = Path(bidsdir).resolve()
    outputdir = Path(outputdir).resolve() if outputdir else bidsdir/'derivatives/fmriprep'

    # The FreeSurfer license is mandatory for fmriprep
    license = license or os.getenv('FS_LICENSE', '')
    if not license or not Path(license).is_file():
        print(f"ERROR: FreeSurfer license file not found: '{license}' (use -l or set $FS_LICENSE)")
        exit(1)
    license = Path(license).resolve()

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

        if not workroot:
            scratch = os.getenv('SCRATCH') or f"/cluster/scratch/{os.getenv('USER')}"
            if not scratch:
                print("ERROR: $SCRATCH is not set")
                exit(1)
            workdir = Path(scratch)/'fmriprep_work'/sub_id
        else:
            workdir = Path(workroot)/sub_id

        done, nconfounds, nbold, crashes = is_done(bidsdir, outputdir, sub_id)
        print(f"\n>>> Found {nconfounds}/{nbold} preprocessed bold runs for: {sub_id}")
        if crashes:
            print(f"    NB: {len(crashes)} crash-file(s) found in: {outputdir/sub_id/'log'}")
        if nbold == 0:
            print(f"--> No func/*_bold data found, skipping job ({n+1}/{len(subjects)}): {sub_id}")
            continue

        # Submit the fmriprep job to the cluster
        if force or not done:

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
                binds += f" --bind {Path(templateflow).resolve()}:/templateflow"
                envs   = '--env TEMPLATEFLOW_HOME=/templateflow'

            # Leave some memory headroom for fmriprep, so that slurm does not kill the job
            mem_mb = int(mem_gb * 1024 * 0.9)

            # Generate the submit-command
            mem_per_cpu = -(-mem_gb * 1024 // nthreads)
            submit = f"sbatch --job-name={jobname} --cpus-per-task={nthreads} --mem-per-cpu={mem_per_cpu} --time={walltime}:00:00 {qargs}"

            # Generate the fmriprep-job
            job = textwrap.dedent(f"""\
                        #!/bin/bash
                        source {envscript}
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
            if skip and jobname in running:
                print(f"--> Skipping already running / scheduled job ({n+1}/{len(subjects)}): {jobname}")
            else:
                print(f"--> {'Running' if nosub else 'Submitting'} job ({n+1}/{len(subjects)}):\n{command}")
                if not dryrun:
                    process = subprocess.run(command, shell=True, capture_output=True, text=True)
                    if process.returncode != 0:
                        print(f"ERROR {process.returncode}: Job {'run' if nosub else 'submission'} failed\n{process.stderr}\n{process.stdout}")
                    elif not nosub:
                        print(process.stdout.strip())

        else:
            print(f"--> Nothing to do for job ({n+1}/{len(subjects)}): {sub_id}")

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
    parser.add_argument('-i','--image',         help='Path to the fMRIPrep apptainer image', default='/cluster/work/gess/sbs/neurodesk/neurocommand/local/containers/fmriprep_25.2.5_20260413/fmriprep_25.2.5_20260413.simg')
    parser.add_argument('-e','--envscript',     help='Path to the environment setup script to source in each job', default='/cluster/work/gess/sbs/MoviefMRI/code/environment.sh')
    parser.add_argument('-l','--license',       help='Path to the FreeSurfer license.txt file (default = $FS_LICENSE)', default='$HOME/FSLicense/license.txt')
    parser.add_argument('--templateflow',       help='Path to a pre-fetched TemplateFlow directory (needed if compute nodes have no internet access)', default='/cluster/work/gess/sbs/MoviefMRI/templateflow')
    parser.add_argument('-w','--workdir',       help='The root working-directory where intermediate files are stored (default = $SCRATCH/fmriprep_work)', default='')
    parser.add_argument('-p','--participants',  help='Space separated list of selected sub-# names / labels to be processed. Otherwise all subjects in the bidsfolder will be selected', nargs='+')
    parser.add_argument('-c','--clobber',       help='If this flag is given subjects will be processed with a clean working directory and their existing fmriprep output removed. Otherwise finished subjects will be skipped', action='store_true')
    parser.add_argument('-r','--resubmit',      help='If this flag is given then jobs are submitted, regardless of already running or scheduled jobs with the same name, otherwise job submission is skipped', action='store_false')
    parser.add_argument('-m','--mem_gb',        help='Required amount of memory in GB (total per job)', default=32, type=int)
    parser.add_argument('-n','--nthreads',      help='Number of CPUs per job', default=8, type=int)
    parser.add_argument('-t','--time',          help='Required walltime in hours', default=24, type=int)
    parser.add_argument('-q','--qargs',         help='Additional (opaque) arguments that are passed to the resource manager (NB: Use quotes and include at least one space character to prevent overearly parsing)', type=str, default='')
    parser.add_argument('--nosub',              help='Add this flag to run the fmriprep commands locally, without submitting them (useful for debugging)', action='store_true')
    parser.add_argument('-d','--dryrun',        help='Add this flag to just print the fmriprep sbatch commands without actually submitting them (useful for debugging)', action='store_true')

    # Parse only what we know
    args, passthrough = parser.parse_known_args()

    main(bidsdir      = args.bidsdir,
         outputdir    = args.outputdir,
         image        = args.image,
         envscript    = args.envscript,
         workroot     = args.workdir,
         license      = args.license,
         templateflow = args.templateflow,
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
