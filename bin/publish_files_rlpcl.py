import glob, json, os
D = "/data/user_data/mbronars/packages/guided_playback/rlpcl_site/data"
print(json.dumps({f"data/{os.path.basename(p)}": p for p in sorted(glob.glob(D + "/*.js"))}))
