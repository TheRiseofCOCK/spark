import os, json, subprocess, sys, time, urllib.request, base64

PAT = globals().get("PAT")      # injected by the worker cell
REPO = globals().get("REPO")
RELEASE_ID = "453282824"

# ================= 1. download the data =================
os.makedirs("/content/data", exist_ok=True)

def api(path, raw=False):
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}",
        headers={"Authorization": f"token {PAT}",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def download(asset_name, dest):
    rel = api(f"releases/{RELEASE_ID}")
    for a in rel["assets"]:
        if a["name"] == asset_name:
            req = urllib.request.Request(a["url"],
                headers={"Authorization": f"token {PAT}",
                         "Accept": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
            open(dest, "wb").write(data)
            print(f"downloaded {asset_name} ({len(data)} bytes)", flush=True)
            return True
    print(f"ASSET NOT FOUND: {asset_name}", flush=True)
    return False

sys.path.insert(0, "/content/data")
download("interpreter.py", "/content/data/interpreter.py")
from interpreter import make_executable, run_tests
download("compose_c1_candidates.json", "/content/data/candidates.json")
os.makedirs("/content/data/adapter", exist_ok=True)
download("adapter_model.safetensors", "/content/data/adapter/adapter_model.safetensors")
download("adapter_config.json", "/content/data/adapter/adapter_config.json")
print("data ready", flush=True)

# ================= 2. load the scaffold =================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(model, "/content/data/adapter")
model.eval()
print("scaffold loaded", flush=True)

cands = json.load(open("/content/data/candidates.json"))
print(f"cases: {len(cands)}", flush=True)

# ================= 3. score each candidate =================
def score_candidate(prompt_text, cand_body):
    """The scaffold's mean token log-likelihood of the patched body given the prompt."""
    text = prompt_text.rstrip("\n") + "\n" + cand_body
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    prompt_len = tok(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        logits = out.logits[:, :-1]
        targets = enc["input_ids"][:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        body_lp = tok_lp[:, prompt_len - 1:]
    return body_lp.mean().item()

def run_repaired(prompt_text, body, test_text, fn):
    return bool(run_tests(make_executable(prompt_text, body, test_text, fn)))

results = {"cases": [], "composed": 0, "graph_atomic": 0, "n": 0,
           "shuffled": 0, "trivial": 0}
import random
random.seed(20261121)

t0 = time.time()
for idx, r in enumerate(cands):
    c_list = r["candidates"]
    if not c_list:
        results["cases"].append({"task_id": r["task_id"], "composed": False,
                                 "note": "no candidates"})
        continue
    scores = [score_candidate(r["prompt"], c) for c in c_list]
    best = c_list[max(range(len(scores)), key=lambda i: scores[i])]
    repaired = run_repaired(r["prompt"], best, r["test"], r["fn"])
    # the graph's own accepted output (the atomic)
    acc = r.get("accepted", "")
    acc_ok = (not acc.strip().startswith("ABSTAIN")) and \
             run_repaired(r["prompt"], acc, r["test"], r["fn"])
    # the shuffled control: a random candidate
    shuf = random.choice(c_list)
    shuf_ok = run_repaired(r["prompt"], shuf, r["test"], r["fn"])
    # the trivial control: the buggy body
    try:
        triv_ok = run_repaired(r["prompt"], r["buggy_body"], r["test"], r["fn"])
    except Exception:
        triv_ok = False
    results["cases"].append({"task_id": r["task_id"], "composed": repaired,
                             "graph_atomic": acc_ok, "shuffled": shuf_ok,
                             "trivial": triv_ok, "n_cands": len(c_list)})
    results["composed"] += repaired
    results["graph_atomic"] += acc_ok
    results["shuffled"] += shuf_ok
    results["trivial"] += triv_ok
    results["n"] += 1
    if (idx + 1) % 25 == 0:
        print(f"[{idx+1}/{len(cands)}] composed={results['composed']} "
              f"graph_atomic={results['graph_atomic']} "
              f"({time.time()-t0:.0f}s)", flush=True)

n = max(results["n"], 1)
verdict = {
    "experiment": "E_COMPOSE_1_C1",
    "n": n,
    "composed_rate": results["composed"] / n,
    "graph_atomic_rate": results["graph_atomic"] / n,
    "scaffold_published_rate": 0.7616,
    "shuffled_control_rate": results["shuffled"] / n,
    "trivial_control_rate": results["trivial"] / n,
    "floor_composed": 0.7616,
    "kill_below": 0.5253,
    "survives": None,
}
verdict["survives"] = verdict["composed_rate"] >= verdict["floor_composed"]
print(json.dumps(verdict, indent=1), flush=True)
os.makedirs(f"/content/outputs/compose_c1", exist_ok=True)
json.dump(verdict, open("/content/outputs/compose_c1/verdict.json", "w"),
          indent=1)
json.dump(results["cases"], open("/content/outputs/compose_c1/cases.json", "w"),
          indent=1)
print("C1_DONE", flush=True)
