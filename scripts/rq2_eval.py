"""RQ2: Belebele accuracy per spelling condition, scored by log-likelihood (no sampling).

    python -X utf8 scripts/rq2_eval.py --model unsloth/gemma-3-4b-it [--conditions all] [--limit N] [--load-4bit]
Writes results/rq2/<model>/<condition>.jsonl (one row per question) and
results/logs/rq2_<model>_<time>.json (versions, revision, GPU, prompt hash, accuracy per condition).

The prompt template is English and byte-identical across conditions; only the Uzbek passage,
question and options change. The model's answer is the letter A-D whose token has the highest
next-token log-probability after "Answer:". One question per forward pass (no padding), so a
result never depends on what else was in the batch.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conditions import CONDITIONS  # noqa: E402
from progress import log as progress  # noqa: E402

SEED = 42
LETTERS = ["A", "B", "C", "D"]
TEMPLATE = ("Read the passage and answer the question with the letter of the correct option.\n\n"
            "Passage: {flores_passage}\n\nQuestion: {question}\n"
            "A. {mc_answer1}\nB. {mc_answer2}\nC. {mc_answer3}\nD. {mc_answer4}\n\nAnswer:")


# Robustness prompts. "chat": the model's own chat template, answer token right after the assistant
# header. "uz": the frame itself in Uzbek, written in the spelling being tested (latin_okina or new2026).
CHAT_TEMPLATE = TEMPLATE[: -len("\n\nAnswer:")] + "\n\nAnswer with only the letter of the correct option."
UZ_TEMPLATE = ("Matnni {read} va savolga {right} javob harfi bilan javob bering.\n\n"
               "Matn: {flores_passage}\n\nSavol: {question}\n"
               "A. {mc_answer1}\nB. {mc_answer2}\nC. {mc_answer3}\nD. {mc_answer4}\n\nJavob:")
UZ_WORDS = {"latin_okina": {"read": "o\u02bbqing", "right": "to\u02bbg\u02bbri"},
            "new2026": {"read": "\u00f6qing", "right": "t\u00f6\u011fri"}}


def letter_ids(tok, prefix=" "):
    """Token id of the letters A-D. With the raw prompts they follow 'Answer:', so with a leading
    space; with a chat template they open the assistant turn, so without one."""
    ids = []
    for l in LETTERS:
        enc = tok.encode(prefix + l, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"' {l}' is {len(enc)} tokens for this tokenizer: {enc}")
        ids.append(enc[0])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="belebele_uzn")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder to apply on top of the model")
    ap.add_argument("--prompt", choices=["raw", "chat", "uz"], default="raw")
    ap.add_argument("--skip-done", action="store_true", help="skip conditions whose result file is already complete")
    args = ap.parse_args()

    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    conds = CONDITIONS if args.conditions == "all" else args.conditions.split(",")
    revision = HfApi().model_info(args.model).sha
    progress(f"[rq2] {args.model} @ {revision[:7]}: downloading (if needed) and loading")
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, **kw).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    ids = letter_ids(tok, "" if args.prompt == "chat" else " ")

    progress(f"[rq2] {args.model}: loaded, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    # --limit runs go to their own folder so a test never overwrites a full run
    tag = (args.model.replace("/", "__") + ("__4bit" if args.load_4bit else "")
           + (f"__{Path(args.adapter).parent.name.split('__')[-1]}" if args.adapter else "")
           + (f"__{args.prompt}prompt" if args.prompt != "raw" else "")
           + (f"__limit{args.limit}" if args.limit else ""))
    out = ROOT / "results/rq2" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": args.model, "revision": revision,
           "load_4bit": args.load_4bit, "adapter": args.adapter, "prompt": args.prompt, "dataset": args.dataset, "limit": args.limit, "seed": SEED,
           "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(), "letter_token_ids": ids,
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "cuda": torch.version.cuda},
           "gpu": torch.cuda.get_device_name(0), "accuracy": {}, "seconds": {}}

    logdir = ROOT / "results/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"rq2_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    for c in conds:
        rows = [json.loads(l) for l in (ROOT / "data/parallel" / args.dataset / f"{c}.jsonl")
                .read_text(encoding="utf-8").splitlines()][: args.limit]
        done = out / f"{c}.jsonl"
        if args.skip_done and done.exists() and len(done.read_text(encoding="utf-8").splitlines()) == len(rows):
            progress(f"[rq2] {tag} {c}: already complete, skipping")
            continue
        t0, correct = time.time(), 0
        with open(out / f"{c}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if args.prompt == "chat":
                    prompt = tok.apply_chat_template([{"role": "user", "content": CHAT_TEMPLATE.format(**r)}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
                    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                elif args.prompt == "uz":
                    enc = tok(UZ_TEMPLATE.format(**r, **UZ_WORDS[c]), return_tensors="pt").to("cuda")
                else:
                    enc = tok(TEMPLATE.format(**r), return_tensors="pt").to("cuda")  # adds BOS where the model expects it
                with torch.no_grad():
                    logits = model(**enc).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)[ids].tolist()
                pred = max(range(4), key=lambda k: lp[k]) + 1
                gold = int(r["correct_answer_num"])
                correct += pred == gold
                f.write(json.dumps({"i": i, "link": r["link"], "question_number": r["question_number"],
                                    "gold": gold, "pred": pred, "logprobs": lp,
                                    "prompt_tokens": enc["input_ids"].shape[1]}) + "\n")
                if (i + 1) % 100 == 0:
                    progress(f"[rq2] {tag} {c}: {i + 1}/{len(rows)}")
        log["accuracy"][c] = correct / len(rows)
        log["seconds"][c] = round(time.time() - t0, 1)
        progress(f"[rq2] {tag} {c:15s} DONE acc={correct / len(rows):.4f}  n={len(rows)}  {log['seconds'][c]}s")
        # saved after every condition, so a crash never loses the record of finished work
        logfile.write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
