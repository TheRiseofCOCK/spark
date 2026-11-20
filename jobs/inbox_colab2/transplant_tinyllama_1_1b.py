import os, json, subprocess, sys, tempfile, time, urllib.request, base64, zipfile

# release any residual CUDA memory from a previous failed job in this kernel
import torch as _torch, gc as _gc
_gc.collect()
_torch.cuda.empty_cache()

# the Colab peft/torchao mismatch: drop torchao so peft takes the default path
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True)

PAT = globals().get("PAT")      # injected by the worker cell
REPO = globals().get("REPO")    # injected by the worker cell
RELEASE_ID = "453106377"
# ---- EDIT THESE TWO LINES PER JOB ----
BASE = "TinyLlama/TinyLlama_v1.1"
NAME = "tinyllama_1_1b"

# ================= 1. download the data =================
os.makedirs("/content/data", exist_ok=True)

def api(method, path, headers=None):
    h = {"Authorization": f"token {PAT}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}",
                                 method=method, headers=h)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()

rel = json.loads(api("GET", f"releases/{RELEASE_ID}").decode())
assets = {a["name"]: a["id"] for a in rel["assets"]}
print("release assets:", list(assets), flush=True)

for fname in ["train_A.jsonl", "heldout_B.jsonl", "humaneval_tasks.zip"]:
    dest = f"/content/data/{fname}"
    if os.path.exists(dest):
        continue
    raw = api("GET", f"releases/assets/{assets[fname]}",
              {"Accept": "application/octet-stream"})
    with open(dest, "wb") as f:
        f.write(raw)
    print(f"downloaded {fname} ({len(raw)} bytes)", flush=True)

with zipfile.ZipFile("/content/data/humaneval_tasks.zip") as z:
    z.extractall("/content/data")
print("data ready:", os.listdir("/content/data"), flush=True)

# ================= 2. the eval apparatus =================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/content/data"
MAX_LEN = 1200
SEED = 20261120
ABSTAIN = "ABSTAIN:"

train_pairs = [json.loads(l) for l in open(f"{DATA}/train_A.jsonl")]
heldout = [json.loads(l) for l in open(f"{DATA}/heldout_B.jsonl")]
print(f"train={len(train_pairs)} heldout={len(heldout)}", flush=True)

def entry_fn(prompt_text):
    import ast
    for node in ast.parse(prompt_text).body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None

def make_executable(prompt_text, body, test_text, fn):
    return prompt_text.rstrip() + "\n" + body + "\n\n" + test_text + f"\ncheck({fn})\n"

def run_tests(code, timeout_s=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout_s)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)

def extract_body(generated):
    lines = []
    for line in generated.splitlines():
        if line.strip().startswith(("def check", "check(", "METADATA")):
            break
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"

def build_input(r):
    return (r["prompt"].rstrip() + "\n" + r["buggy_body"].rstrip() +
            "\n\nTest failure:\n" + r["failure_output"].strip() +
            "\n\nPassing body:\n")

OUT = f"/content/outputs/transplant_{NAME}"
os.makedirs(OUT, exist_ok=True)

def save_results(tag, extra):
    payload = {"base": BASE, "name": NAME, "tag": tag, **extra}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(payload, f, indent=2)

def eval_model(model, tok, tag, limit=0):
    cases = heldout[:limit] if limit else heldout
    tok.padding_side = "left"
    inps = [build_input(r) for r in cases]
    texts = []
    bs = 2 if "1_1b" in NAME or "1.1b" in BASE.lower() else 8
    model.eval()
    for j in range(0, len(inps), bs):
        chunk = inps[j:j + bs]
        enc = {k: v.cuda() for k, v in tok(
            chunk, return_tensors="pt", add_special_tokens=False,
            padding=True).items()}
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=512, do_sample=False,
                               pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id)
        for k in range(len(chunk)):
            texts.append(tok.decode(o[k][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True))
        if (j // bs) % 10 == 0:
            print(f"  [{tag}] {j+len(chunk)}/{len(inps)}", flush=True)
    results = []
    for r, text in zip(cases, texts):
        btxt = extract_body(text)
        abstain = btxt.strip().startswith(ABSTAIN)
        repaired = False
        if not abstain:
            test_text = open(f"{DATA}/humaneval_tasks/{r['task_id']}.test.py").read()
            repaired = bool(run_tests(make_executable(
                r["prompt"], btxt, test_text, entry_fn(r["prompt"]))))
        results.append({"task_id": r["task_id"], "bug_class": r["bug_class"],
                        "abstain": abstain, "repaired": repaired})
    n = len(results)
    rate = sum(1 for x in results if x["repaired"]) / n
    verdict = {"n": n, "repair_rate": round(rate, 4)}
    save_results(tag, {"verdict": verdict, "cases": results})
    print(f"[{tag}] repair_rate={rate:.4f} (n={n})", flush=True)
    return rate

# ================= 3. the prompted baseline (secondary gate) =================
print("=== loading base model for the prompted baseline ===", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda()
prompted_rate = eval_model(model, tok, "prompted_baseline")
del model
torch.cuda.empty_cache()

# ================= 4. re-form with the recipe (LoRA r=16, attention+MLP) =================
print("=== re-forming with the recipe (LoRA r=16) ===", flush=True)
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

class BodyEmitter(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        for r in pairs:
            inp = build_input(r)
            self.examples.append((inp, r["fixed_body"]))
        self.tok = tokenizer
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        inp, out = self.examples[i]
        full = inp + out + "<|endoftext|>"
        enc = self.tok(full, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        labels = enc["input_ids"].clone()
        p = self.tok(inp, return_tensors="pt")["input_ids"].shape[1]
        labels[:, :p] = -100
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0], "labels": labels[0]}

def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True)
    mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True,
                          padding_value=-100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}

torch.manual_seed(SEED)
if "pythia" in BASE.lower():
    # GPTNeoX: fused QKV + attention/MLP dense layers
    TARGETS = ["query_key_value", "dense", "dense_h_to_4hf", "dense_4h_to_h"]
else:
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                 target_modules=TARGETS, task_type="CAUSAL_LM")
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).cuda()
model.gradient_checkpointing_enable()   # T4 OOM guard for >0.5B targets
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

ds = BodyEmitter(train_pairs, tok)
BATCH = 2 if "1.1b" in BASE.lower() or "1_1b" in NAME.lower() else 4
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
for ep in range(3):
    tot, n = 0.0, 0
    for step, batch in enumerate(dl):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        tot += out.loss.item()
        n += 1
    print(f"epoch={ep} mean_loss={tot/max(n,1):.4f}", flush=True)
    save_results("training", {"epochs_done": ep + 1, "last_loss": tot/max(n,1)})

model.save_pretrained(f"{OUT}/adapter")
print("adapter saved", flush=True)

# ================= 5. the re-formed eval (primary gate) =================
reformed_rate = eval_model(model, tok, "reformed")

# ================= 6. verdict =================
FLOOR = 0.7616   # the source scaffold's declared floor
verdict = {
    "base": BASE, "name": NAME, "source_floor": FLOOR,
    "prompted_baseline": prompted_rate, "reformed": reformed_rate,
    "survives_floor": reformed_rate >= FLOOR,
    "beats_prompted": reformed_rate > prompted_rate,
}
save_results("verdict", verdict)
print(json.dumps(verdict, indent=2), flush=True)
