# gemma4-e2b-engram

[DeepSeek's Engram](https://github.com/deepseek-ai/Engram) (conditional n-gram memory) bolted onto **Gemma 4 E2B**, and also **Gemma 3** (tested on 270M). Attached with PyTorch hooks, so no Transformers source is modified, and the output is **exactly the original model until you train the Engram weights**.

## How it works

At selected decoder layers, the last 2 and 3 tokens are hashed (multiplicative-XOR, per-layer prime moduli, several heads) into learned embedding tables. The retrieved vector is gated by the layer's hidden state (`sigmoid(sqrt-signed(norm(k)·norm(q)/√d))`), passed through a dilated depthwise causal conv, and added to the residual stream before the layer runs.

Changes from the upstream demo: single residual stream (Gemma has no hyper-connections); hashing done in torch on-device; KV-cache-aware (n-gram history and conv state carry across decode steps); zero-initialised output projection; hook-based attachment.

## Install & use

```bash
pip install -e ".[dev]"
python -m pytest -q          # 8 offline tests, no downloads
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma4_engram import EngramConfig, attach_engram, freeze_base, save_engram

tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", dtype=torch.bfloat16)
attach_engram(model, tok, EngramConfig(layer_ids=[1, 15]))   # E2B has 35 layers
emb_params, other_params = freeze_base(model)                # train only Engram
```

Train / generate:

```bash
python scripts/train.py --model google/gemma-4-E2B --data corpus.txt --layers 1 15 --out engram.pt
python examples/generate.py --model google/gemma-4-E2B --engram engram.pt --prompt "Hello"
```

Engram weights live in a separate file (`save_engram` / `load_engram`), not in the base checkpoint. Defaults add ~140M parameters on E2B (~3%); tune with `table_size`, `n_embed_per_ngram`, `n_head_per_ngram`, `layer_ids`.

## Results so far

One CPU run: Gemma 3 270M (frozen), Engram on layers 1 and 9 (9.1M params), WikiText-2 train, 250 steps × 4 × 128 tokens (~128k tokens, ~17 min, 2 CPU cores). Held-out loss on the last 5% of the corpus (same domain, 32 batches):

| | val loss | perplexity |
|---|---|---|
| Frozen base | 3.987 | 53.9 |
| + Engram (trained) | 3.829 | 46.0 |

Log and checkpoint: [`runs/`](runs).

**Read this with caution.** It is a single seed on a tiny, in-domain dataset, with no control. Part of the gain may be adaptation to WikiText's formatting rather than "memory" (a control with frozen random tables, or an equal-sized LoRA, is needed to say). It shows the plumbing works and that training reduces loss. It is **not** evidence that Engram improves Gemma. **Gemma 4 E2B has not been run**: the code is tested on a shrunken Gemma 4 config (offline tests) but not on the real 10 GB checkpoint.

## Notes

- Gemma needs a BOS token at position 0. `scripts/train.py` adds one to every block. Without it the baseline perplexity is ~8× worse.
- Inputs passed as `inputs_embeds` (no `input_ids`) skip Engram, since it needs token IDs to hash. Multimodal image/audio placeholder tokens are hashed like ordinary tokens.
- `google/gemma-3-270m` is gated on Hugging Face; the run above used the ungated mirror `unsloth/gemma-3-270m`.
- Upstream's README says Engram *models* are under a separate Model License; the code is Apache-2.0. Gemma weights follow the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Credits

Engram: DeepSeek-AI, [paper & code](https://github.com/deepseek-ai/Engram). See `NOTICE`.
