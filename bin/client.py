"""POST a guided request to a running server: python bin/client.py http://host:8766/guide '{"domain":"pickup",...}'"""
import json, sys, urllib.request
url, body = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "{}"
r = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"}, method="POST" if url.rstrip("/").split("/")[-1] in ("guide", "generate", "plan") else "GET")
print(json.dumps(json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read()), indent=1))
