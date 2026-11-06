"""Train only the Engram parameters on a frozen Gemma 3 / Gemma 4 model, with held-out evaluation.

Examples
  # Gemma 3 270M, CPU-friendly smoke run
  python scripts/train.py --model unsloth/gemma-3-270m --data data.txt --layers 1 9 --steps 40 --seq-len 128 --batch 4
  # Gemma 4 E2B on a GPU
  python scripts/train.py --model google/gemma-4-E2B --data data.txt --layers 1 15

--data is a plain text file; the last --val-frac of the tokens is held out and evaluated before and after training.
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma4_engram import EngramConfig, attach_engram, freeze_base, save_engram

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-4-E2B")
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="engram.pt")
ap.add_argument("--layers", type=int, nargs="+", default=[1, 15])
ap.add_argument("--table-size", type=int, default=131072)
ap.add_argument("--n-embed", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--seq-len", type=int, default=1024)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-4)        # projections / conv / norms
ap.add_argument("--emb-lr", type=float, default=1.5e-3)  # memory tables (paper: larger LR, no weight decay)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--val-batches", type=int, default=16)
ap.add_argument("--max-train-tokens", type=int, default=0, help="0 = use all")
ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log", default="")
args = ap.parse_args()
torch.manual_seed(args.seed)

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(args.dtype, torch.bfloat16 if dev == "cuda" else torch.float32)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev)
cfg = EngramConfig(layer_ids=args.layers, table_size=args.table_size, n_embed_per_ngram=args.n_embed, n_head_per_ngram=args.heads)
attach_engram(model, tok, cfg, cache_dir=".")
emb, other = freeze_base(model)
print(f"engram params: {sum(p.numel() for p in emb + other)/1e6:.2f}M | base params: {sum(p.numel() for n,p in model.named_parameters() if not n.startswith('engram'))/1e6:.0f}M")

ids = tok(open(args.data, encoding="utf-8").read(), return_tensors="pt", add_special_tokens=False).input_ids[0]
cut = int(ids.numel() * (1 - args.val_frac))
def pack(x):
    # Gemma needs BOS at position 0 (attention sink); each block = [BOS] + (seq_len-1) tokens
    body = args.seq_len - 1
    n = x.numel() // body
    blocks = x[: n * body].view(n, body)
    bos = torch.full((n, 1), tok.bos_token_id, dtype=blocks.dtype)
    return torch.cat([bos, blocks], dim=1)
train, val = pack(ids[:cut]), pack(ids[cut:])
if args.max_train_tokens:
    train = train[: args.max_train_tokens // args.seq_len]
print(f"train blocks {len(train)} | val blocks {len(val)} | seq {args.seq_len}")

@torch.no_grad()
def evaluate():
    model.eval()
    g = torch.Generator().manual_seed(123)
    idx = torch.randperm(len(val), generator=g)[: args.val_batches * args.batch]
    tot, n = 0.0, 0
    for i in range(0, len(idx), args.batch):
        b = val[idx[i:i + args.batch]].to(dev)
        tot += model(input_ids=b, labels=b).loss.item() * len(b); n += len(b)
    model.train()
    return tot / n

opt = torch.optim.AdamW([{"params": emb, "lr": args.emb_lr, "weight_decay": 0.0},
                         {"params": other, "lr": args.lr, "weight_decay": 0.01}], betas=(0.9, 0.95))
warm = max(1, args.steps // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.1, 1 - s / args.steps))

t0 = time.time(); v0 = evaluate()
print(f"[before] val loss {v0:.4f}  ppl {2.718281828 ** v0:.2f}  ({time.time()-t0:.0f}s)", flush=True)
hist = []
model.train()
for step in range(args.steps):
    b = train[torch.randint(0, len(train), (args.batch,))].to(dev)
    loss = model(input_ids=b, labels=b).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(emb + other, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    hist.append(loss.item())
    if step % 5 == 0 or step == args.steps - 1:
        print(f"step {step:4d} loss {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
v1 = evaluate()
print(f"[after]  val loss {v1:.4f}  ppl {2.718281828 ** v1:.2f}  delta {v1 - v0:+.4f}")
save_engram(model, args.out)
print("saved", args.out)
if args.log:
    json.dump({"args": vars(args), "val_before": v0, "val_after": v1, "train_loss": hist}, open(args.log, "w"))
