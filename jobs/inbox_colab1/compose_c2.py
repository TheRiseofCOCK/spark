import os, json, subprocess, sys, time, urllib.request, base64, random

PAT = globals().get("PAT")
REPO = globals().get("REPO")
RELEASE_ID = "453282824"

import torch as _torch, gc as _gc
_gc.collect()
_torch.cuda.empty_cache()

os.makedirs("/content/data", exist_ok=True)

def api(path):
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
download("heldout_B.jsonl", "/content/data/heldout_B.jsonl")
download("adapter_model.safetensors", "/content/data/adapter/adapter_model.safetensors")
download("adapter_config.json", "/content/data/adapter/adapter_config.json")
print("data ready", flush=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(model, "/content/data/adapter")
model.eval()
print("scaffold loaded", flush=True)

cands = json.load(open("/content/data/candidates.json"))
# re-attach the failure outputs (the candidates JSON omits them)
_heldout = {json.loads(l)["task_id"]: json.loads(l)
            for l in open("/content/data/heldout_B.jsonl")}
for r in cands:
    r["failure_output"] = _heldout.get(r["task_id"], {}).get("failure_output", "")
K = 8

def build_prompt(r):
    # the scaffold's training prompt format (the transplant job's build_input)
    return (r["prompt"].rstrip() + "\n" + r["buggy_body"].rstrip() +
            "\n\nTest failure:\n" + r.get("failure_output", "").strip() +
            "\n\nPassing body:\n")

def gen_candidates(r, k):
    """The scaffold's greedy output + (k-1) temperature samples."""
    prompt = build_prompt(r)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    outs = []
    with torch.no_grad():
        greedy = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(greedy[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True))
        for _ in range(k - 1):
            s = model.generate(**enc, max_new_tokens=512, do_sample=True,
                               temperature=0.8, top_p=0.95,
                               pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(s[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs

def extract_body(text):
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("```"):
            return s
    return text.strip()

results = {"composed": 0, "greedy_only": 0, "shuffled": 0, "n": 0,
           "oracle_top3": 0, "cases": []}
random.seed(20261121)

t0 = time.time()
for idx, r in enumerate(cands):
    try:
        raw_cands = gen_candidates(r, K)
    except Exception as e:
        print(f"[{idx}] gen error: {e}", flush=True)
        continue
    c_list = [extract_body(c) for c in raw_cands]
    # dedupe while preserving order
    seen, uniq = set(), []
    for c in c_list:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    c_list = uniq[:K]

    composed = greedy_ok = shuf_ok = False
    top3_hit = None
    for i, c in enumerate(c_list):
        ok = bool(run_tests(make_executable(r["prompt"], c, r["test"], r["fn"])))
        if i == 0:
            greedy_ok = ok
        if ok and composed is False:
            composed = True
            if top3_hit is None:
                top3_hit = i < 3
    if not composed:
        composed = False
    shuf = random.choice(c_list)
    shuf_ok = bool(run_tests(make_executable(r["prompt"], shuf, r["test"], r["fn"])))

    results["composed"] += composed
    results["greedy_only"] += greedy_ok
    results["shuffled"] += shuf_ok
    if composed and top3_hit:
        results["oracle_top3"] += 1
    results["n"] += 1
    results["cases"].append({"task_id": r["task_id"], "composed": composed,
                             "greedy": greedy_ok, "shuffled": shuf_ok,
                             "n_cands": len(c_list)})
    if (idx + 1) % 25 == 0:
        print(f"[{idx+1}/{len(cands)}] composed={results['composed']} "
              f"greedy={results['greedy_only']} ({time.time()-t0:.0f}s)",
              flush=True)

n = max(results["n"], 1)
verdict = {
    "experiment": "E_COMPOSE_1_C2",
    "n": n,
    "composed_rate": results["composed"] / n,
    "greedy_only_rate": results["greedy_only"] / n,
    "scaffold_published_rate": 0.7616,
    "shuffled_control_rate": results["shuffled"] / n,
    "oracle_top3_share": results["oracle_top3"] / max(results["composed"], 1),
    "floor_composed": 0.7616,
    "kill_below": 0.7616,
}
verdict["survives"] = verdict["composed_rate"] >= verdict["floor_composed"]
print(json.dumps(verdict, indent=1), flush=True)
os.makedirs("/content/outputs/compose_c2", exist_ok=True)
json.dump(verdict, open("/content/outputs/compose_c2/verdict.json", "w"), indent=1)
json.dump(results["cases"], open("/content/outputs/compose_c2/cases.json", "w"), indent=1)
print("C2_DONE", flush=True)
