#!/bin/zsh
# footprint of the training process every 5 minutes
while true; do
  PID=$(pgrep -f "python -u runs/mac_2b/run_mps.py train" | head -1)
  [ -z "$PID" ] && break
  echo "$(date -u +%H:%M:%S) $(footprint $PID | grep 'Footprint:' | sed 's/.*Footprint: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')" >> runs/mac_2b/full/mem.txt
  python3 -c "import time; time.sleep(300)"
done
