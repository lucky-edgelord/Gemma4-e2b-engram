import numpy as np
import torch

from gemma4_engram import (CompressedTokenizer, EngramConfig, NgramHasher, attach_engram,
                           freeze_base, load_engram, save_engram)

CFG = EngramConfig(layer_ids=[1, 3], table_size=2003, n_embed_per_ngram=32, n_head_per_ngram=2)


def test_token_compression_merges_case_and_space(tok):
    comp = CompressedTokenizer(tok)
    lt = comp.lookup_table
    assert lt[0] == lt[1] == lt[2] == lt[3]          # tokN, TOKN, " tokN", Tokn collapse
    assert lt[0] != lt[4]
    assert comp.num_new_token == VOCAB_GROUPS


VOCAB_GROUPS = 512 // 4


def test_hash_ranges_and_determinism(tok):
    h = NgramHasher(CFG, CompressedTokenizer(tok), pad_token_id=0)
    ids = torch.randint(0, 512, (2, 20))
    full = torch.cat([torch.zeros(2, 2, dtype=torch.long), h.compress(ids)], 1)
    a, b = h.hash(full, 1), h.hash(full, 1)
    assert torch.equal(a, b) and a.shape == (2, 20, 4)
    mods = torch.tensor(h.mods[1])
    assert (a >= 0).all() and (a < mods).all()
    assert len(set(h.mods[1]) & set(h.mods[3])) == 0   # primes are unique across layers


def test_hash_depends_only_on_preceding_tokens(tok):
    h = NgramHasher(CFG, CompressedTokenizer(tok), pad_token_id=0)
    x = h.compress(torch.randint(0, 512, (1, 12)))
    y = x.clone(); y[0, 9] = (y[0, 9] + 7) % 128          # change a later token
    pre = torch.zeros(1, 2, dtype=torch.long)
    ha, hb = h.hash(torch.cat([pre, x], 1), 1), h.hash(torch.cat([pre, y], 1), 1)
    assert torch.equal(ha[:, :9], hb[:, :9])              # causal: earlier positions unchanged
    assert not torch.equal(ha[:, 9], hb[:, 9])


def test_zero_init_is_exact_noop(tiny_model, tok):
    ids = torch.randint(1, 512, (2, 24))
    with torch.no_grad():
        ref = tiny_model(ids).logits
    attach_engram(tiny_model, tok, CFG)
    with torch.no_grad():
        out = tiny_model(ids).logits
    assert torch.allclose(ref, out, atol=1e-5)


def _activate(model):
    for b in model.engram.values():
        torch.nn.init.normal_(b.value_proj.weight, std=0.05)


def test_engram_changes_output_once_weights_nonzero(tiny_model, tok):
    ids = torch.randint(1, 512, (1, 24))
    with torch.no_grad():
        ref = tiny_model(ids).logits
    attach_engram(tiny_model, tok, CFG)
    _activate(tiny_model)
    with torch.no_grad():
        assert not torch.allclose(ref, tiny_model(ids).logits, atol=1e-3)


def test_cached_generation_matches_uncached(tiny_model, tok):
    attach_engram(tiny_model, tok, CFG)
    _activate(tiny_model)
    ids = torch.randint(1, 512, (2, 10))
    with torch.no_grad():
        a = tiny_model.generate(ids, max_new_tokens=10, do_sample=False, use_cache=True)
        b = tiny_model.generate(ids, max_new_tokens=10, do_sample=False, use_cache=False)
    assert torch.equal(a, b)


def test_only_engram_trains(tiny_model, tok):
    attach_engram(tiny_model, tok, CFG)
    emb, other = freeze_base(tiny_model)
    _activate(tiny_model)
    tiny_model.train()
    ids = torch.randint(1, 512, (2, 16))
    tiny_model(ids, labels=ids).loss.backward()
    assert emb[0].grad is not None and emb[0].grad.abs().sum() > 0
    assert all(p.grad is None for n, p in tiny_model.named_parameters() if not n.startswith("engram"))


def test_save_load_roundtrip(tiny_model, tok, tmp_path):
    attach_engram(tiny_model, tok, CFG)
    _activate(tiny_model)
    path = tmp_path / "e.pt"
    save_engram(tiny_model, str(path))
    ids = torch.randint(1, 512, (1, 16))
    with torch.no_grad():
        want = tiny_model(ids).logits
    torch.manual_seed(0)
    from transformers import Gemma4ForCausalLM
    from conftest import tiny_config
    fresh = Gemma4ForCausalLM(tiny_config()).eval()
    fresh.load_state_dict({k: v for k, v in tiny_model.state_dict().items() if not k.startswith("engram")})
    load_engram(fresh, tok, str(path))
    with torch.no_grad():
        assert torch.allclose(want, fresh(ids).logits, atol=1e-5)
