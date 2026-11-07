#!/usr/bin/env python3
"""
mriqc_sub.py is a wrapper around mriqc that queries the bids directory for new
participants and then runs participant-level mriqc jobs on the compute cluster.
If the participant-level jobs have all finished, you can run the group-level
"mriqc bidsdir outputdir group" command to generate group level results (the
group level report and the features CSV table)

"""

import os
import shutil
import subprocess
import tempfile
import argparse
import textwrap
import shlex
from pathlib import Path


def main(bidsdir: str, outputdir: str, image: str, envscript: str, workroot: str, sessions=(), force=False, mem_gb=18, walltime=8, args='', qargs='', dryrun=False, nosub=False, skip=True):

    bidsdir   = Path(bidsdir)
    outputdir = Path(outputdir)

    # Default
    if not outputdir.name:
        outputdir = bidsdir/'derivatives/mriqc'

    # Map the bids session-directories
    if not sessions:
        sessions = list(bidsdir.glob('sub-*/ses-*'))
        if not sessions:
            sessions = list(bidsdir.glob('sub-*'))      # Try without session-subfolders
    else:
        sessions = [bidsdir/session for session in sessions]

    # Loop over bids session-directories and submit a job for every (new) session
    for n, session in enumerate(sessions):

        if not session.is_dir():
            print(f">>> Directory does not exist: {session}")
            continue

        niifiles = list(session.glob('*/*.nii*'))
        missing = [f for f in niifiles if not f.exists()]
        if missing:
            print(f">>> Skipping {session}: {len(missing)} file(s) not downloaded")
            continue

        sub_id = [part for part in session.parts if part.startswith('sub-')][0]
        ses_id = [part for part in session.parts if part.startswith('ses-')]
        if ses_id:
            ses_id     = ses_id[0]
            ses_id_opt = f" --session-id {ses_id[4:]}"
        else:
            ses_id     = ''
            ses_id_opt = ''

        if not workroot:
            scratch = os.getenv('SCRATCH')
            if not scratch:
                print("ERROR: $SCRATCH is not set")
                exit(1)
            workdir = Path(scratch)/'mriqc_work'/f"{sub_id}_{ses_id}"
        else:
            workdir = Path(workroot)/f"{sub_id}_{ses_id}"

        # A session is considered already done if there are html-reports for every anat/*_T?w and every func/*_bold file
        nrniifiles = len(list((bidsdir/sub_id/ses_id/'anat')      .glob(f"{sub_id}_{ses_id}*T?w.nii*")))  + \
                     len(list((bidsdir/sub_id/ses_id/'extra_data').glob(f"{sub_id}_{ses_id}*T?w.nii*")))  + \
                     len(list((bidsdir/sub_id/ses_id/'func')      .glob(f"{sub_id}_{ses_id}*bold.nii*"))) + \
                     len(list((bidsdir/sub_id/ses_id/'extra_data').glob(f"{sub_id}_{ses_id}*bold.nii*")))
        reports    = list(outputdir.glob(f"{sub_id}_{ses_id}*.html"))
        print(f"\n>>> Found {len(reports)}/{nrniifiles} existing MRIQC-reports for: {sub_id}_{ses_id}")

        # Submit the mriqc job to the cluster
        if force or not len(reports)==nrniifiles:

            # Start with a clean directory if we are forcing to reprocess the data (as presumably something went wrong or has changed)
            if force:
                if workdir.is_dir():
                    print(f"Cleaning: {workdir}")
                    if not dryrun:
                        shutil.rmtree(workdir, ignore_errors=True)          # NB: This can also be done in parallel on the cluster if it takes too much time
                for report in reports:
                    print(f"Cleaning: {report}")
                    if not dryrun:
                        report.unlink()

            # Generate the submit-command
            submit  = f"sbatch --job-name=mriqc_{sub_id}_{ses_id} --mem-per-cpu={mem_gb}G --time={walltime}:00:00 {qargs}"
            running = subprocess.run('squeue -u $USER -o format=%j | grep mriqc_sub', shell=True, capture_output=True, text=True)

            # Generate the mriqc-job
            job = textwrap.dedent(f"""\
                        #!/bin/bash
                        source {envscript}
                        echo "$neurodesk_singularity_opts"
                        ulimit -v unlimited
                        cd {Path.cwd()}
                        mkdir -p {workdir}/tmp
                        apptainer exec --cleanenv --bind {workdir}/tmp:/tmp {image} mriqc {bidsdir} {outputdir} participant -w {workdir} --participant-label {sub_id[4:]} {ses_id_opt} --verbose-reports --mem_gb {mem_gb} --ants-nthreads 1 --nprocs 1 {args}""")

            # Submit the job to the compute cluster
            if nosub:
                workdir.mkdir(parents=True, exist_ok=True)
                command = job
            else:
                command = f"{submit} <<EOF\n{job}\nEOF"
            if skip and f"mriqc_{sub_id}_{ses_id}" in running.stdout:
                print(f"--> Skipping already running / scheduled job ({n+1}/{len(sessions)}): mriqc_{sub_id}_{ses_id}")
            else:
                print(f"--> {'Running' if nosub else 'Submitting'} job ({n+1}/{len(sessions)}):\n{command}")
                if not dryrun:
                    process = subprocess.run(command, shell=True, capture_output=True, text=True)
                    if process.stderr or process.returncode!=0:
                        print(f"ERROR {process.returncode}: Job submission failed\n{process.stderr}\n{process.stdout}")
                    if nosub:
                        shutil.rmtree(workdir, ignore_errors=True)

        else:
            print(f"--> Nothing to do for job ({n+1}/{len(sessions)}): {session}")

    if not sessions:
        print(f"No BIDS subject/session folders found in {bidsdir}")
    elif dryrun:
        print('\n----------------\nDone! NB: The printed jobs were not actually submitted')
    elif nosub:
        print('\n----------------\nDone!')
    else:
        print('\n----------------\n'
                f"Done! Now wait for the jobs to finish... Check that e.g. with this command:\n\n  squeue -u {os.getenv('USER')} -o %j | grep mriqc_sub\n\n"
                'When finished you can run e.g. a group-level QC analysis like this:\n\n'
                f"  mriqc_group {bidsdir}\n\n")


# Shell usage
if __name__ == "__main__":

    # Parse the input arguments and run bidscoiner(args)
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter, description=textwrap.dedent(__doc__),
                                     epilog='Any unrecognized options are forwarded directly to mriqc\n\n'
                                            'for more information see:\n'
                                            '  module help mriqc\n'
                                            '  mriqc -h\n\n'
                                            'examples:\n'
                                            '  mriqc_sub /project/3022026.01/bids\n'
                                            '  mriqc_sub /project/3022026.01/bids -w /project/3022026.01/mriqc_work\n'
                                            '  mriqc_sub /project/3022026.01/bids -o /project/3022026.01/mriqc --sessions sub-010/ses-mri01 sub-011/ses-mri01\n'
                                            '  mriqc_sub /project/3022026.01/bids --fft-spikes-detector --no-sub -q "-N 1 -c 4 --ntasks-per-node=1"\n'
                                            '  mriqc_sub -f -m 16 /project/3022026.01/bids -s sub-013/ses-mri01\n\n'
                                            'Author:\n' 
                                            '  Marcel Zwiers\n '
                                            'adapted by Luca A. Naudszus\n')
    parser.add_argument('bidsdir',          help='The bids-directory with the subject data')
    parser.add_argument('-o','--outputdir', help='The mriqc output-directory where the html-reports will be stored (default = bidsdir/derivatives/mriqc)', default='')
    parser.add_argument('-i','--image', help='Path to the MRIQC apptainer .sif image', default='/cluster/work/gess/sbs/neurodesk/neurocommand/local/containers/mriqc_24.0.2_20241108/mriqc_24.0.2_20241108.simg')
    parser.add_argument('-e','--envscript', help='Path to the environment setup script to source in each job', default='/cluster/work/gess/sbs/MoviefMRI/code/environment.sh')
    parser.add_argument('-w','--workdir',   help='The working-directory where intermediate files are stored (default = a temporary directory', default='')
    parser.add_argument('-s','--sessions',  help='Space separated list of selected sub-#/ses-# names / folders to be processed. Otherwise all sessions in the bidsfolder will be selected', nargs='+')
    parser.add_argument('-c','--clobber',   help='If this flag is given subjects will be processed with a clean working directory, regardless of existing folders in the bidsfolder. Otherwise existing folders will be skipped', action='store_true')
    parser.add_argument('-r','--resubmit',  help='If this flag is given then job are submitted, regardless of already running or scheduled jobs with the same name, otherwise job submission is skipped', action='store_false')
    parser.add_argument('-m','--mem_gb',    help='Required amount of memory in GB', default=18, type=int)
    parser.add_argument('-t','--time',      help='Required walltime in hours', default=8, type=int)
    parser.add_argument('-q','--qargs',     help='Additional (opaque) arguments that are passed to the resource manager (NB: Use quotes and include at least one space character to prevent overearly parsing)', type=str, default='')
    parser.add_argument('-n','--nosub',     help='Add this flag to run the mriqc commands locally, without submitting them (useful for debugging)', action='store_true')
    parser.add_argument('-d','--dryrun',    help='Add this flag to just print the mriqc sbatch commands without actually submitting them (useful for debugging)', action='store_true')

    # Parse only what we know
    args, passthrough = parser.parse_known_args()

    main(bidsdir   = args.bidsdir,
         outputdir = args.outputdir,
         image     = args.image,
         envscript = args.envscript,
         workroot  = args.workdir,
         sessions  = args.sessions,
         force     = args.clobber,
         mem_gb    = args.mem_gb,
         walltime  = args.time,
         args      = " ".join(shlex.quote(a) for a in passthrough),
         qargs     = args.qargs,
         dryrun    = args.dryrun,
         nosub     = args.nosub,
         skip      = args.resubmit)
