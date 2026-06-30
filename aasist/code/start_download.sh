#!/bin/bash
# Tiny launcher: starts the 2021 download in the background, prints PID.
# Survives shell disconnect via nohup.
mkdir -p "/scratch/$USER/aasist/data"
LOGFILE="/scratch/$USER/aasist/data/download_2021.log"
SCRIPT="/scratch/$USER/aasist/code/aasist_branch/download_2021.sh"
nohup bash "$SCRIPT" > "$LOGFILE" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true
echo "Download started"
echo "PID:  $PID"
echo "Log:  $LOGFILE"
echo "Tail: tail -5 $LOGFILE"
