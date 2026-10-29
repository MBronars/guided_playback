cd /data/user_data/mbronars/packages/guided_playback
until curl -s -m 5 http://babel-q9-28:8768/ | grep -q '"busy"'; do sleep 15; done
echo "server 0 up $(date +%H:%M)"; bash sweeps/resume_fresh_q1.sh 10566183
