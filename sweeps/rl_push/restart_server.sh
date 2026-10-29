#!/bin/bash
# wait for the in-flight request (rlpA_ep008) to finish, then restart the server with GP_RELOAD=1 (objective edits reload per request)
cd /data/user_data/mbronars/packages/guided_playback
until grep -q "wrote .*rlpA_ep008" logs/serve_rlpush_8797.log; do sleep 5; done
sleep 3; kill 2653583; sleep 5; pgrep -f "serve.py.*8797" >/dev/null && sleep 5
mv logs/serve_rlpush_8797.log logs/serve_rlpush_8797.run1.log
nohup sweeps/rl_push/serve_rlpush.sh > logs/serve_rlpush_8797.log 2>&1 < /dev/null &
until grep -q "\[serve\] http" logs/serve_rlpush_8797.log 2>/dev/null; do sleep 5; done
echo SERVER_READY
