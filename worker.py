#!/usr/bin/env python3
"""The tile-factory worker: every pod runs this. No SSH deployments —
work arrives as job specs in jobs/*.json (delivered by sync.sh), claims go
through the HF ledger lock, artifacts go to the vault, health goes to
pods/<name>.json. Push to the repo = deploy to the fleet.

Job spec (jobs/<id>.json):
  { "id": str,                  # unique; ledger tracks completed[id]
    "ready": "bash test",       # exit 0 = this pod can run it now (optional)
    "script": "bash script",    # the work; exit 0 = success
    "timeoutSec": 7200 }        # optional
"""
import base64, json, os, socket, subprocess, time, datetime, glob, sys
import urllib.request

REPO = os.environ.get("HF_REPO_ID", "miguelemosreverte/alambique-datasets")
SCENE = "world/kazbek-c10"
HERE = os.path.dirname(os.path.abspath(__file__))

def _token():
    return os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

def _commit(ops, summary, timeout=60):
    """Raw commit API with a HARD timeout — a stalled socket must fail loudly,
    never hang the worker into silent uselessness."""
    body = [json.dumps({"key": "header", "value": {"summary": summary}})]
    for op in ops:
        body.append(json.dumps(op))
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/commit/main",
        data="\n".join(body).encode(),
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/x-ndjson"},
        method="POST")
    urllib.request.urlopen(req, timeout=timeout)


def pod_name():
    explicit = os.environ.get("POD_NAME")
    if explicit:
        return explicit
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.split("\n")[0]
        gpu = gpu.strip().replace("NVIDIA ", "").replace(" ", "-")[:24]
    except Exception:
        pass
    return f"{gpu or 'cpu'}-{socket.gethostname()[:10]}"


ME = pod_name()
UTC = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{UTC()}] {msg}", flush=True)


def fetch(path):
    """Fresh small-file read from the vault (no cache)."""
    try:
        import urllib.request
        tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        req = urllib.request.Request(
            f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}",
            headers={"Authorization": f"Bearer {tok}"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception:
        return None


def put(path, obj, summary):
    _commit([{"key": "file", "value": {"path": path,
              "content": base64.b64encode(json.dumps(obj, indent=2).encode()).decode(),
              "encoding": "base64"}}], summary)


def rm(path, summary):
    _commit([{"key": "deletedFile", "value": {"path": path}}], summary)


def with_lock(mutate, summary):
    """Ledger lock protocol: write-then-confirm, TTL steal, mutate, release."""
    lock_path = f"{SCENE}/ledger.lock"
    lock = fetch(lock_path)
    if lock:
        age = time.time() - datetime.datetime.fromisoformat(
            lock["acquiredAt"].replace("Z", "+00:00")).timestamp()
        if age < lock.get("ttlSeconds", 300):
            return False  # someone holds it; try next cycle
        log(f"stealing expired lock from {lock.get('holder')}")
    put(lock_path, {"holder": ME, "acquiredAt": UTC(), "ttlSeconds": 300}, f"lock: {ME}")
    time.sleep(5)
    confirm = fetch(lock_path)
    if not confirm or confirm.get("holder") != ME:
        return False  # lost the race
    try:
        ledger = fetch(f"{SCENE}/ledger.json") or {}
        mutate(ledger)
        ledger["updatedAt"] = UTC()
        put(f"{SCENE}/ledger.json", ledger, summary)
    finally:
        try:
            rm(lock_path, f"unlock: {ME}")
        except Exception:
            pass
    return True


def run_job(job):
    jid = job["id"]

    def claim(ledger):
        ledger.setdefault("claims", {})[jid] = {"holder": ME, "claimedAt": UTC(), "status": "running"}

    if not with_lock(claim, f"{ME} claims {jid}"):
        return
    log(f"claimed {jid}; running")
    open("/tmp/worker-busy", "w").write(jid)
    try:
        r = subprocess.run(["bash", "-lc", job["script"]], timeout=job.get("timeoutSec", 7200),
                           capture_output=True, text=True)
        ok = r.returncode == 0
        tail = (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        ok, tail = False, "TIMEOUT"
    log(f"{jid}: {'OK' if ok else 'FAILED'}\n{tail[-500:]}")

    def finish(ledger):
        ledger.setdefault("claims", {}).pop(jid, None)
        key = "completed" if ok else "failed"
        ledger.setdefault(key, {})[jid] = {"holder": ME, "at": UTC(), "tail": tail[-400:]}

    for _ in range(20):  # finishing the books must not be lost to lock contention
        if with_lock(finish, f"{jid} {'done' if ok else 'FAILED'} on {ME}"):
            break
        time.sleep(30)
    try:
        os.remove("/tmp/worker-busy")
    except FileNotFoundError:
        pass
    if ok and job.get("_queue_path"):
        try:
            rm(job["_queue_path"], f"queue request served by {ME}")
        except Exception:
            pass


def list_queue():
    """Tile-generation requests awaiting a builder (best-effort listing)."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{REPO}/tree/main/generation-queue",
            headers={"Authorization": f"Bearer {_token()}"})
        entries = json.loads(urllib.request.urlopen(req, timeout=30).read())
        out = []
        for e in entries[:8]:
            body = fetch(e["path"])
            if body:
                body["_path"] = e["path"]
                out.append(body)
        return out
    except Exception:
        return []


def queue_job(jid, vault, req):
    """A tile request means the vault's octree is missing pieces: rebuild the
    whole octree from its model PLY and upload — supersets any single leaf."""
    scene_dir = "/".join(vault.split("/")[:-1])          # world/<scene>
    ver = vault.split("/")[-1]                            # octree-vN
    script = f"""set -e
TOK=$(cat ~/.cache/huggingface/token)
LEDGER=$(curl -sfL -H "Authorization: Bearer $TOK" "https://huggingface.co/datasets/{REPO}/resolve/main/{scene_dir}/ledger.json?cb=$RANDOM")
PLY=$(echo "$LEDGER" | python3 -c "import json,sys; L=json.load(sys.stdin); print(next((m.get('ply') for m in L.get('models',{{}}).values() if m.get('ply')), ''))")
test -n "$PLY" || {{ echo "no PLY in ledger for {vault}"; exit 1; }}
curl -sfL -H "Authorization: Bearer $TOK" -o /tmp/queue-model.ply "https://huggingface.co/datasets/{REPO}/resolve/main/$PLY"
export DENO_INSTALL=$HOME/.deno
command -v $DENO_INSTALL/bin/deno >/dev/null 2>&1 || (curl -fsSL https://deno.land/install.sh | sh -s -- -y >/dev/null 2>&1)
STAGE=/workspace/queue-stage/{scene_dir}
mkdir -p $STAGE
$DENO_INSTALL/bin/deno run --allow-read --allow-write /workspace/image-generation/splat-viewer/build-octree.ts /tmp/queue-model.ply $STAGE/{ver}
D=$(ls -d $STAGE/*octree* | head -1); [ "$D" = "$STAGE/{ver}" ] || mv "$D" $STAGE/{ver}
python3 -c "from huggingface_hub import HfApi; HfApi().upload_large_folder(folder_path='/workspace/queue-stage', repo_id='{REPO}', repo_type='dataset', print_report=False)"
echo QUEUE-{ver}-REBUILT"""
    return {{"id": jid, "script": script, "timeoutSec": 3600,
             "_queue_path": req.get("_path")}}


def main():
    log(f"worker up as {ME}")
    last_beat = 0.0
    cycles = 0
    not_ready: dict[str, float] = {}
    while True:
        try:
            cycles += 1
            if cycles % 10 == 1:
                log(f"cycle {cycles}: scanning jobs")
            if time.time() - last_beat > 900:
                put(f"pods/{ME}.json", {"pod": ME, "aliveAt": UTC(),
                    "commit": subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()},
                    f"heartbeat {ME}")
                last_beat = time.time()
            ledger = fetch(f"{SCENE}/ledger.json") or {}
            done = set(ledger.get("completed", {})) | set(ledger.get("failed", {}))
            claimed = set()
            for cid, c in (ledger.get("claims") or {}).items():
                try:
                    age = time.time() - datetime.datetime.fromisoformat(
                        c["claimedAt"].replace("Z", "+00:00")).timestamp()
                    if age < c.get("claimTtlSec", 5400):
                        claimed.add(cid)
                    else:
                        log(f"claim {cid} by {c.get('holder')} expired ({age/3600:.1f} h) — eligible again")
                except Exception:
                    claimed.add(cid)
            # the lazy-world loop: runtime files tile requests into
            # generation-queue/; any worker turns them into octree rebuilds.
            for req in list_queue():
                vault = req.get("vault", "")
                jid = f"queue-{vault.replace('/', '_')}"
                if jid in done or jid in claimed:
                    continue
                run_job(queue_job(jid, vault, req))
                break
            for jf in sorted(glob.glob(f"{HERE}/jobs/*.json")):
                job = json.load(open(jf))
                jid = job["id"]
                if jid in done or jid in claimed:
                    continue
                if time.time() < not_ready.get(jid, 0):
                    continue
                if job.get("ready"):
                    probe = subprocess.run(["bash", "-lc", job["ready"]], capture_output=True)
                    if probe.returncode != 0:
                        not_ready[jid] = time.time() + 120
                        continue
                run_job(job)
                break  # one job per cycle; re-read the ledger before the next
        except Exception as e:
            log(f"cycle error: {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
