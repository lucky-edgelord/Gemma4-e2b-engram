import pytest
import torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM

VOCAB = 512


class StubTokenizer:
    """Minimal tokenizer surface used by Engram; ids 0..511, with deliberate case/space collisions."""
    pad_token_id = 0

    def __len__(self):
        return VOCAB

    def decode(self, ids, skip_special_tokens=False):
        i = ids[0]
        base = f"tok{i // 4}"                     # 4 ids share a base word
        return [base, base.upper(), " " + base, base.capitalize()][i % 4]

    def convert_ids_to_tokens(self, i):
        return f"<{i}>"


def tiny_config(**over):
    d = dict(
        vocab_size=VOCAB, vocab_size_per_layer_input=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        global_head_dim=32, hidden_size_per_layer_input=16, num_kv_shared_layers=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention",
                     "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=16, max_position_embeddings=256,
    )
    d.update(over)
    return Gemma4TextConfig(**d)


@pytest.fixture
def tok():
    return StubTokenizer()


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return Gemma4ForCausalLM(tiny_config()).eval()
