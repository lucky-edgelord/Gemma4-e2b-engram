"""Load a base model, attach trained Engram weights, and generate.

python examples/generate.py --model unsloth/gemma-3-270m --engram runs/engram_gemma3_270m.pt --prompt "The history of"
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma4_engram import load_engram

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engram", default="", help="path from scripts/train.py; omit for the plain base model")
ap.add_argument("--prompt", default="The history of")
ap.add_argument("--max-new-tokens", type=int, default=60)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
if args.engram:
    load_engram(model, tok, args.engram, cache_dir=".")
ids = tok(args.prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
