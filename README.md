# spark - the Fossil Labs GPU relay

Command relay between the Fossil Labs orchestrator (spark1/spark2, LAN) and remote
GPU workers (Google Colab notebooks).

## Protocol (v1)
- jobs/inbox_colab1/ and jobs/inbox_colab2/ - one .py file per job
- A worker claims a job by deleting its inbox file, runs it on the Colab VM, then
  writes jobs/results/<worker>/<job_id>.out.txt (stdout/stderr + status) and
  uploads any files the job wrote to /content/outputs/<job_id>/ to
  jobs/results/<worker>/<job_id>/
- The orchestrator places jobs and reads results. Workers never touch each other's
  inboxes.
