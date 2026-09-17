#!/usr/bin/env bash
# Read-only queue checks and scheduler estimate; never submits/cancels real jobs.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: bash cluster/check_queue.sh PARTITION [JOB_ID]\n' >&2
  exit 2
fi
partition="$1"
if [[ "$partition" == -* ]]; then
  printf 'Invalid partition name\n' >&2
  exit 2
fi
sinfo -p "$partition" -o '%P %a %l %D %t %G'
squeue --me -o '%.18i %.14P %.22j %.10T %.10M %.10l %R'
sbatch --test-only --partition="$partition" cluster/probe_gpu.slurm
if [[ $# == 2 ]]; then
  if [[ ! "$2" =~ ^[0-9]+$ ]]; then
    printf 'JOB_ID must be numeric\n' >&2
    exit 2
  fi
  squeue --start --jobs="$2"
  sacct -j "$2" --format=JobID,JobName,Partition,State,Submit,Start,End,Elapsed,Timelimit,ExitCode
fi
