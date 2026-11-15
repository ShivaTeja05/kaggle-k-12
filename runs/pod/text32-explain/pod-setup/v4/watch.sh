#!/bin/sh
# Touch the dead-man heartbeat while the 32B pod job runs; exit when its status file appears (max 11 h).
cd "$(dirname "$0")/../../../../.."
ADDR=$(cat runs/pod/text32-explain/pod-setup/v4/addr.txt)
SSH="ssh -o ConnectTimeout=20 -i $HOME/.runpod/ssh/runpodctl-ssh-key -p ${ADDR#* } root@${ADDR% *}"
end=$(( $(date +%s) + 39600 ))
while [ $(date +%s) -lt $end ]; do
  touch runs/heartbeat
  out=$($SSH 'cat /workspace/runs/text32-v4.status 2>/dev/null' 2>/dev/null)
  if [ -n "$out" ]; then echo "STATUS: $out"; break; fi
  sleep 180
done
touch runs/heartbeat
$SSH 'tail -3 /workspace/runs/run.log; tr "\r" "\n" < /workspace/runs/text32-v4.log | grep -E "map3|test_scored|COMPLETE|Error|error" | tail -8; ls /workspace/runs/text32-v4'
