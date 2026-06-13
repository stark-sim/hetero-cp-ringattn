#!/bin/bash
PID=$(pgrep -f "run_a100_serial_vs_overlap.sh" | head -1)
LOG="/tmp/a100_ab_nohup.log"
while true; do
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "STATUS: COMPLETED"
    echo "=== last 50 lines ==="
    tail -n 50 "$LOG" 2>/dev/null
    break
  fi
  MODE=$(grep -oE "\[(SERIAL|PIPELINE)\]" "$LOG" 2>/dev/null | tail -1)
  echo "STATUS: RUNNING mode=${MODE:-unknown}"
  echo "=== last 15 lines ==="
  tail -n 15 "$LOG"
  sleep 30
done
