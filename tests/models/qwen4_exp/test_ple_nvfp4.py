"""NVFP4 PLE sidecar parsing, RAM/UVA and direct-disk lookup."""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.ple_nvfp4 import (
    Nvfp4DiskRowTable, PinnedNVFP4Table, prefer_pinned_ple, resolve_nvfp4_source,
)

from .common import EOS, hash_constants, requires_cuda, toy_hf_config
from .test_ple import _meta
from .test_ple_disk import _embedding

pytest.importorskip("freetoken.kernel._ple_store")


def _write_sidecar(folder, packed, scales, shards):
    from safetensors.torch import save_file

    folder.mkdir()
    per = packed.shape[0] // shards
    for i in range(shards):
        rows = slice(i * per, (i + 1) * per)
        save_file({
            "weight_e2m1": packed[rows],
            "weight_scale": scales[rows],
            "weight_scale_2": torch.tensor(0.25 * (i + 1)),
        }, str(folder / f"shard_{i}.safetensors"))
    (folder / "META.json").write_text(json.dumps({
        "layout": "group16_e2m1_e4m3scale_lownibblefirst",
        "shards": shards, "rows": packed.shape[0], "width": packed.shape[1] * 2,
    }), encoding="utf-8")


def _reference(packed, scales, scale_2, rows_per_shard, ids):
    ids = ids.cpu().reshape(-1)
    selected = packed[ids]
    codes = torch.stack((selected & 0xF, selected >> 4), dim=-1).flatten(1).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    block_scales = scales[ids].float().repeat_interleave(16, dim=1)
    globals_ = torch.tensor(scale_2)[ids // rows_per_shard, None]
    return (lut[codes] * block_scales * globals_).to(torch.bfloat16)


@requires_cuda
def test_nvfp4_sidecar_ram_disk_and_policy(tmp_path):
    args = parse_config(toy_hf_config()).qwen4_args
    multipliers, sizes, offsets = hash_constants(args)
    total_rows = int(offsets[-1] + sizes[-1])
    shards = next(n for n in (4, 2, 1) if total_rows % n == 0)
    gen = torch.Generator().manual_seed(53)
    packed = torch.randint(0, 256, (total_rows, args.ngram_head_dim // 2),
                           dtype=torch.uint8, generator=gen)
    scales = torch.randn(total_rows, args.ngram_head_dim // 16,
                         generator=gen).to(torch.float8_e4m3fn)
    _write_sidecar(tmp_path / "ples_nvfp4", packed, scales, shards)
    source = resolve_nvfp4_source(str(tmp_path), str(tmp_path),
                                  expected_rows=total_rows, expected_width=args.ngram_head_dim)
    assert source is not None and len(source.scale_2) == shards

    tokens = [3, 4, EOS, 5, 9]
    ids = _embedding().row_ids(_meta([tokens], [[EOS, EOS]])).cuda()
    want = _reference(packed, scales, source.scale_2, source.packed.rows_per_extent, ids)
    constants = {
        "num_ngram_heads": args.num_ngram_heads,
        "layer_multipliers": multipliers.tolist(),
        "per_head_vocab_sizes": sizes.tolist(),
        "per_head_offsets": offsets.tolist(),
        "eos_token_id": EOS,
    }
    disk = Nvfp4DiskRowTable(source, constants)
    disk.fill([torch.tensor([EOS, EOS, *tokens])], graph=False)
    assert torch.equal(disk.lookup(ids).cpu().view_as(want), want)
    out = torch.empty(len(tokens), args.num_ngram_heads * args.ngram_head_dim,
                      dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            disk.lookup(ids, out)
    torch.cuda.current_stream().wait_stream(stream)
    disk.fill([torch.tensor([EOS, EOS, *tokens])], graph=True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu().view_as(want), want)

    from freetoken.moe.host_banks import HostBank
    packed_bytes = packed.numel()
    bank = HostBank((packed_bytes + scales.numel(),), torch.uint8)
    bank_packed = bank.tensor[:packed_bytes].view_as(packed).copy_(packed)
    bank_scales = bank.tensor[packed_bytes:].view(torch.float8_e4m3fn).view_as(scales).copy_(scales)
    bank.pin()
    ram = PinnedNVFP4Table(bank_packed, bank_scales, source.scale_2)
    assert torch.equal(ram.lookup(ids).cpu().view_as(want), want)

    gib = 1 << 30
    assert prefer_pinned_ple(27 * gib, 64 * gib, (192 * gib, 180 * gib))
    assert not prefer_pinned_ple(27 * gib, 64 * gib, (96 * gib, 90 * gib))
