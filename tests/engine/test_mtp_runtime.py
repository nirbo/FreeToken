from types import SimpleNamespace

import pytest
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


def test_pinned_ple_mtp_uses_overlap(monkeypatch):
    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        model_config=SimpleNamespace(num_speculative_tokens=4), ple_backend="pinned"
    )
    scheduler.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: scheduler.stream)
    monkeypatch.setattr(ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    scheduler.overlap_loop = lambda _: (_ for _ in ()).throw(RuntimeError("overlap"))

    with pytest.raises(RuntimeError, match="overlap"):
        scheduler.run_forever()


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


def test_speculative_prefill_uses_decode_moe_path(monkeypatch):
    from freetoken.layers.moe import OffloadMoELayer

    layer = object.__new__(OffloadMoELayer)
    layer.tp_size = 1
    calls = []
    monkeypatch.setattr(
        "freetoken.layers.moe.get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=True, speculative=True)),
    )
    monkeypatch.setattr(
        layer, "decode_forward", lambda hidden, router: calls.append("decode") or hidden
    )
    monkeypatch.setattr(
        layer, "prefill_forward", lambda hidden, router: calls.append("prefill") or hidden
    )

    hidden = torch.zeros(2, 3)
    assert layer.forward(hidden) is hidden
    assert calls == ["decode"]


def test_rejected_speculative_pages_return_to_free_list():
    from freetoken.scheduler.cache import CacheManager

    manager = object.__new__(CacheManager)
    manager.page_size = 64
    manager.page_table = torch.arange(256, dtype=torch.int32).view(1, -1)
    manager.free_slots = torch.empty(0, dtype=torch.int32)
    manager.swa_pool = None
    manager.swa_paged = False
    req = SimpleNamespace(table_idx=0, cached_len=128, device_len=129)

    manager.free_speculative_tail(req, allocated_len=130)

    assert manager.free_slots.tolist() == [128]
