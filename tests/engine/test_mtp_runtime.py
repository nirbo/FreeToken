from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine, _greedy_accept
from freetoken.scheduler.scheduler import _speculative_depth


def test_mtp_round_math_and_budget_cap():
    drafts = torch.tensor([11, 12, 13], dtype=torch.int32)

    accepted, licensed = _greedy_accept(torch.tensor([11, 99, 13, 14]), drafts)
    assert accepted == 1
    assert licensed.tolist() == [11, 99]

    accepted, licensed = _greedy_accept(torch.tensor([11, 12, 13, 14]), drafts)
    assert accepted == 3
    assert licensed.tolist() == [11, 12, 13, 14]

    req = SimpleNamespace(
        remain_len=3, sampling_params=SimpleNamespace(is_greedy=True)
    )
    assert _speculative_depth(req, 5) == 2
    req.sampling_params.is_greedy = False
    assert _speculative_depth(req, 5) == 0


def test_mtp_verification_materializes_drafts_for_disk_ple():
    engine = object.__new__(Engine)
    engine.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    req = SimpleNamespace(input_ids=torch.tensor([10, 20, 99], dtype=torch.int32))
    batch = engine._mtp_batch(
        req,
        torch.tensor([30, 40], dtype=torch.int32),
        torch.tensor([2, 3]),
        torch.tensor([2, 3]),
        cached_len=2,
        phase="prefill",
        host_tokens=True,
    )
    assert batch.reqs[0].input_ids.tolist() == [10, 20, 30, 40]
