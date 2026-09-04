from types import SimpleNamespace

import torch

from freetoken.layers.moe import MoELayer
from freetoken.models.qwen4_exp.weight import _mtp_expert_banks


def test_resident_nvfp4_bank_layout(monkeypatch):
    monkeypatch.setattr("freetoken.layers.moe.get_tp_info", lambda: SimpleNamespace(size=1))
    layer = MoELayer(
        num_experts=3,
        top_k=2,
        hidden_size=32,
        intermediate_size=16,
        weight_format="nvfp4",
    )

    expected = {
        "gate_up_packed": ((3, 32, 16), torch.uint8),
        "gate_up_scale": ((3, 32, 2), torch.float8_e4m3fn),
        "gate_up_global": ((3, 32), torch.float16),
        "down_packed": ((3, 32, 8), torch.uint8),
        "down_scale": ((3, 32, 1), torch.float8_e4m3fn),
        "down_global": ((3, 32), torch.float16),
    }
    state = layer.state_dict()
    assert set(state) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert state[name].shape == shape
        assert state[name].dtype == dtype


def test_mtp_experts_pack_into_the_resident_layout():
    class Reader:
        def get_tensor(self, name):
            expert = int(name.split(".experts.")[1].split(".")[0])
            proj = next(p for p in ("gate_proj", "up_proj", "down_proj") if p in name)
            value = expert * 10 + {"gate_proj": 1, "up_proj": 2, "down_proj": 3}[proj]
            rows, packed_cols = ((32, 8) if proj == "down_proj" else (16, 16))
            if name.endswith("weight_scale_2"):
                return torch.tensor(value, dtype=torch.float32)
            if name.endswith("weight_scale"):
                return torch.full((rows, packed_cols // 8), value, dtype=torch.float8_e4m3fn)
            return torch.full((rows, packed_cols), value, dtype=torch.uint8)

    config = SimpleNamespace(num_experts=2, hidden_size=32, moe_intermediate_size=16)
    banks = _mtp_expert_banks(Reader(), config, torch.device("cpu"))

    assert torch.all(banks["gate_up_packed"][1, :16] == 11)
    assert torch.all(banks["gate_up_packed"][1, 16:] == 12)
    assert torch.all(banks["down_packed"][1] == 13)
    assert torch.all(banks["gate_up_global"][0, :16] == 1)
    assert torch.all(banks["gate_up_global"][0, 16:] == 2)
