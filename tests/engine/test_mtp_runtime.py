from types import SimpleNamespace

import torch

from freetoken.engine.engine import _greedy_accept
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
