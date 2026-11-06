"""
Engram (DeepSeek conditional memory) adapted for Gemma 4 (E2B / E4B ...) and Gemma 3 (e.g. 270M).

Differences from deepseek-ai/Engram `engram_demo_v1.py`:
  * Gemma has a single residual stream, so hyper-connections are dropped (hc_mult = 1).
  * Hashing runs in torch on-device (no numpy round trip) and is cache-aware, so
    `model.generate()` with a KV cache still sees the previous tokens' n-grams.
  * Engram is attached with hooks; no Transformers source is modified.
  * The output projection is zero-initialised, so the patched model is *exactly*
    the original model until you train the Engram parameters.

Usage:
    from gemma4_engram import EngramConfig, attach_engram, save_engram, load_engram
    model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", dtype=torch.bfloat16)
    tok   = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    attach_engram(model, tok, EngramConfig())
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from tokenizers import Regex, normalizers


# --------------------------------------------------------------------------- config
@dataclass
class EngramConfig:
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])  # decoder layers that get Engram
    max_ngram_size: int = 3          # uses 2-grams .. max_ngram_size-grams
    n_embed_per_ngram: int = 256     # total embedding width per n-gram order (split across heads)
    n_head_per_ngram: int = 4        # independent hash heads per n-gram order
    table_size: int = 131072         # approx. rows per head (rounded up to a prime)
    kernel_size: int = 4             # depthwise causal conv (dilation = max_ngram_size)
    seed: int = 0


# --------------------------------------------------------------------------- tokenizer compression
class CompressedTokenizer:
    """Collapses token ids that are equal after NFKC/NFD/accent-strip/lowercase/whitespace norm."""

    def __init__(self, tokenizer, cache_path: Optional[str] = None):
        self.tokenizer = tokenizer
        if cache_path and os.path.exists(cache_path):
            self.lookup_table = np.load(cache_path)
        else:
            self.lookup_table = self._build()
            if cache_path:
                np.save(cache_path, self.lookup_table)
        self.num_new_token = int(self.lookup_table.max()) + 1

    def _build(self) -> np.ndarray:
        sentinel = ""
        norm = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ])
        n = len(self.tokenizer)
        key2new: Dict[str, int] = {}
        lookup = np.empty(n, dtype=np.int64)
        for tid in range(n):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                key = norm.normalize_str(text) or text
            lookup[tid] = key2new.setdefault(key, len(key2new))
        return lookup


# --------------------------------------------------------------------------- hashing
def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not (isprime(c) and c not in seen):
        c += 1
    return c


class NgramHasher:
    def __init__(self, cfg: EngramConfig, comp: CompressedTokenizer, pad_token_id: int):
        self.cfg = cfg
        self.comp = comp
        self.pad = int(comp.lookup_table[pad_token_id])
        v = comp.num_new_token
        half = max(1, (np.iinfo(np.int64).max // v) // 2)
        self.mult: Dict[int, torch.Tensor] = {}
        for lid in cfg.layer_ids:
            g = np.random.default_rng(int(cfg.seed + 10007 * lid))
            r = g.integers(0, half, size=(cfg.max_ngram_size,), dtype=np.int64)
            self.mult[lid] = torch.from_numpy(r * 2 + 1)
        # per layer -> flat list of prime moduli, ordered (ngram 2..N) x heads
        seen: set = set()
        self.mods: Dict[int, List[int]] = {}
        for lid in cfg.layer_ids:
            mods = []
            for _ in range(2, cfg.max_ngram_size + 1):
                start = cfg.table_size - 1
                for _ in range(cfg.n_head_per_ngram):
                    start = _next_prime(start, seen)
                    seen.add(start)
                    mods.append(start)
            self.mods[lid] = mods
        self._lookup = torch.from_numpy(comp.lookup_table)

    def compress(self, ids: torch.Tensor) -> torch.Tensor:
        return self._lookup.to(ids.device)[ids.clamp_min(0)]

    def hash(self, comp_ids: torch.Tensor, lid: int) -> torch.Tensor:
        """comp_ids: [B, T] (already prefixed with N-1 history/pad tokens) -> [B, T-(N-1), n_hash]."""
        N = self.cfg.max_ngram_size
        B, L = comp_ids.shape
        T = L - (N - 1)
        m = self.mult[lid].to(comp_ids.device)
        # token k positions back, for the last T positions
        back = [comp_ids[:, (N - 1 - k): (N - 1 - k) + T] for k in range(N)]
        out = []
        mi = 0
        for n in range(2, N + 1):
            mix = back[0] * m[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, back[k] * m[k])
            for _ in range(self.cfg.n_head_per_ngram):
                out.append(mix % self.mods[lid][mi])
                mi += 1
        return torch.stack(out, dim=2)


# --------------------------------------------------------------------------- the module
class EngramBlock(nn.Module):
    """Single-stream Engram: hashed n-gram memory -> context-gated value -> + dilated causal conv."""

    def __init__(self, cfg: EngramConfig, hidden: int, mods: List[int], eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
        offs = np.concatenate([[0], np.cumsum(mods)[:-1]])
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(int(sum(mods)), head_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        in_dim = (cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram
        self.key_proj = nn.Linear(in_dim, hidden, bias=False)
        self.value_proj = nn.Linear(in_dim, hidden, bias=False)
        nn.init.zeros_(self.value_proj.weight)           # => exact no-op at init
        self.norm_k = nn.RMSNorm(hidden, eps=eps)
        self.norm_q = nn.RMSNorm(hidden, eps=eps)
        self.conv_norm = nn.RMSNorm(hidden, eps=eps)
        self.dilation = cfg.max_ngram_size
        self.conv = nn.Conv1d(hidden, hidden, cfg.kernel_size, groups=hidden, bias=False,
                              dilation=self.dilation, padding=0)
        self.pad_len = (cfg.kernel_size - 1) * self.dilation

    def forward(self, hidden_states: torch.Tensor, hashes: torch.Tensor,
                state: Optional[torch.Tensor] = None):
        """state: previous conv inputs [B, pad_len, D] (cached decoding) or None (zeros).
        Returns (output, new_state)."""
        B, T, D = hidden_states.shape
        e = self.embedding(hashes + self.offsets).flatten(2)          # [B,T,in_dim]
        q = self.norm_q(hidden_states.float())
        k = self.norm_k(self.key_proj(e))
        gate = (q * k).sum(-1, keepdim=True) / math.sqrt(D)
        gate = (gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid()
        v = gate * self.value_proj(e)
        u = self.conv_norm(v)
        if state is None or state.shape[0] != B:
            state = u.new_zeros(B, self.pad_len, D)
        u_full = torch.cat([state.to(u.dtype), u], dim=1)                  # causal left context
        y = self.conv(u_full.transpose(1, 2)).transpose(1, 2)              # [B,T,D]
        new_state = u_full[:, -self.pad_len:].detach()
        return (v + F.silu(y)).to(hidden_states.dtype), new_state


class _Ctx:
    """Shared per-forward state: current hashes per layer + token history for KV-cached decoding."""

    def __init__(self, hasher: NgramHasher):
        self.hasher = hasher
        self.tail: Optional[torch.Tensor] = None
        self.hashes: Optional[Dict[int, torch.Tensor]] = None
        self.cached = False                                   # True while decoding with a KV cache
        self.conv_state: Dict[int, torch.Tensor] = {}


# --------------------------------------------------------------------------- attach / save / load
_LAYER_CLASSES = {"Gemma4TextDecoderLayer", "Gemma3DecoderLayer"}


def _find_layers(model) -> Dict[int, nn.Module]:
    layers = {m.layer_idx: m for m in model.modules() if type(m).__name__ in _LAYER_CLASSES}
    if not layers:
        raise ValueError(f"No decoder layer of type {sorted(_LAYER_CLASSES)} found - is this a Gemma 3/4 model?")
    return layers


def attach_engram(model, tokenizer, cfg: EngramConfig = EngramConfig(), cache_dir: Optional[str] = None):
    layers = _find_layers(model)
    for lid in cfg.layer_ids:
        if lid not in layers:
            raise ValueError(f"layer {lid} not in model (has {len(layers)} layers)")

    cache = os.path.join(cache_dir, "engram_token_lookup.npy") if cache_dir else None
    comp = CompressedTokenizer(tokenizer, cache)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    hasher = NgramHasher(cfg, comp, pad)
    ctx = _Ctx(hasher)
    N = cfg.max_ngram_size

    blocks = nn.ModuleDict()
    for lid in cfg.layer_ids:
        layer = layers[lid]
        dev = layer.input_layernorm.weight.device
        blocks[str(lid)] = EngramBlock(cfg, layer.input_layernorm.weight.shape[0], hasher.mods[lid]).to(dev)  # fp32 on purpose
    model.add_module("engram", blocks)
    model._engram_ctx, model._engram_cfg = ctx, cfg

    # 1) outer model: grab input_ids (the text decoder only ever sees inputs_embeds)
    def outer_pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args and torch.is_tensor(args[0]) else None)
        if ids is None or ids.dim() != 2:
            ctx.hashes = None
            return
        pkv = kwargs.get("past_key_values")
        cached = pkv is not None and hasattr(pkv, "get_seq_length") and pkv.get_seq_length() > 0
        c = hasher.compress(ids)
        ctx.cached = bool(cached)
        if not cached:
            ctx.conv_state = {}
        if cached and ctx.tail is not None and ctx.tail.shape[0] == c.shape[0]:
            prefix = ctx.tail.to(c.device)
        else:
            prefix = torch.full((c.shape[0], N - 1), hasher.pad, dtype=c.dtype, device=c.device)
        full = torch.cat([prefix, c], dim=1)
        ctx.tail = full[:, -(N - 1):].detach()
        ctx.hashes = {lid: hasher.hash(full, lid) for lid in cfg.layer_ids}

    model.register_forward_pre_hook(outer_pre, with_kwargs=True)

    # 2) decoder layers: add Engram output to the residual stream going in
    def make_layer_hook(lid):
        blk = blocks[str(lid)]

        def hook(module, args, kwargs):
            if ctx.hashes is None:
                return
            h = args[0] if args else kwargs["hidden_states"]
            hs = ctx.hashes[lid].to(h.device)
            if hs.shape[1] != h.shape[1]:   # e.g. multimodal inputs_embeds path with mismatched length
                return
            out, new_state = blk(h, hs, ctx.conv_state.get(lid) if ctx.cached else None)
            ctx.conv_state[lid] = new_state
            h = h + out
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs, hidden_states=h)
            return args, kwargs

        return hook

    for lid in cfg.layer_ids:
        layers[lid].register_forward_pre_hook(make_layer_hook(lid), with_kwargs=True)
    return model


def freeze_base(model):
    """Freeze everything except Engram; returns (embedding_params, other_engram_params)."""
    for p in model.parameters():
        p.requires_grad_(False)
    emb, other = [], []
    for name, p in model.engram.named_parameters():
        p.requires_grad_(True)
        (emb if "embedding" in name else other).append(p)
    return emb, other


def save_engram(model, path: str):
    torch.save({"cfg": asdict(model._engram_cfg), "state": model.engram.state_dict()}, path)


def load_engram(model, tokenizer, path: str, cache_dir: Optional[str] = None):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    attach_engram(model, tokenizer, EngramConfig(**blob["cfg"]), cache_dir)
    model.engram.load_state_dict(blob["state"])
    return model
