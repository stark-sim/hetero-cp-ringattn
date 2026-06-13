#!/bin/bash
LOG="/tmp/a100_lc_coord.log"
PID_FILE="/tmp/a100_lc_coord.pid"

while true; do
  PID=$(cat "$PID_FILE" 2>/dev/null)
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "STATUS: COMPLETED"
    echo "=== last 30 lines of coordinator log ==="
    tail -n 30 "$LOG" 2>/dev/null
    break
  fi

  PROGRESS=$(grep -oE "Request [0-9]+ / [0-9]+" "$LOG" 2>/dev/null | tail -1)
  echo "STATUS: RUNNING"
  echo "PROGRESS: ${PROGRESS:-unknown}"
  echo "=== last 10 lines ==="
  tail -n 10 "$LOG"
  sleep 30
done
