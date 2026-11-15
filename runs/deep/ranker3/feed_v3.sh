#!/bin/sh
# Runs on this Mac. When stage A ends on the pod: copy Qwen3.8's answers here, build the ranker v3 files and upload them.
# Usage: feed_v3.sh HOST PORT
H=$1; P=$2; cd "$(dirname "$0")/../../.." || exit 1
s() { ssh -i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=15 -p $P root@$H "$@"; }
L=runs/deep/ranker3/feed.log; say() { echo "$(date -u +%H:%M:%S) $*" | tee -a $L; }
say "waiting for stage A"
while :; do case "$(s 'cat /workspace/runs/q38j.status' 2>/dev/null)" in exit*) break;; esac; python3 -c "import time; time.sleep(30)"; done
say "stage A ended: $(s 'cat /workspace/runs/q38j.status')"
for try in 1 2 3; do s 'cd /workspace/runs && tar czf - tr_*.jsonl run_tr_*.log' | tar xzf - -C runs/deep/q38/train_pod && break; python3 -c "import time; time.sleep(10)"; done
say "answers here: $(cat runs/deep/q38/train_pod/tr_*.jsonl | wc -l) lines"
PYTHONPATH=src:scripts ~/.cache/k12-api-venv/bin/python scripts/build_ranker_v3.py --out data/v10 --blind 0.1 > runs/deep/ranker3/build.log 2>&1 || { say "BUILD FAILED"; exit 1; }
say "built: $(wc -l < data/v10/pw_train.jsonl) train rows, $(wc -l < data/v10/pw_dev.jsonl) dev rows"
for try in 1 2 3; do
  COPYFILE_DISABLE=1 tar czf - -C data/v10 pw_train.jsonl pw_dev.jsonl $(cd data/v10 && ls pw_test_*.jsonl) | s 'tar xzf - -C /workspace/data/v10' && \
  [ "$(s 'cat /workspace/data/v10/pw_train.jsonl | wc -c' | tr -d ' ')" = "$(wc -c < data/v10/pw_train.jsonl | tr -d ' ')" ] && { s 'touch /workspace/data/v10/ready'; say "uploaded and marked ready"; exit 0; }
  python3 -c "import time; time.sleep(10)"
done
say "UPLOAD FAILED"; exit 1
