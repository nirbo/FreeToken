# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in pinned
host memory and the GPU dereferences it in place over PCIe -- at its host VA on Linux/UVA, at
the mapped device address on WDDM (``kernel/pinned.device_ptr``). One program per requested
row: read the row, widen to fp32, apply the per-tensor scale, store bf16.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.nvfp4_dequant import _e2m1_lut

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if IS_FP8:
        if e4m3_native_cx():
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0))
    else:
        base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.where(in_range, values * scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            row_ids,
            out,
            float(scale),
            num_rows,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


@triton.jit
def _ple_nvfp4_kernel(
    packed_ptr, scales_ptr, ids_ptr, out_ptr, lut_ptr, scale_2_ptr,
    num_rows, rows_per_shard,
    PACKED_D: tl.constexpr, GATHER: tl.constexpr, BLOCK_D: tl.constexpr,
):
    out_row = tl.program_id(0)
    global_row = tl.load(ids_ptr + out_row).to(tl.int64)
    in_range = (global_row >= 0) & (global_row < num_rows)
    global_row = tl.where(in_range, global_row, 0)
    row = global_row if GATHER else out_row
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < PACKED_D

    packed_base = packed_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    packed = tl.load(packed_base + row * PACKED_D + offsets, mask=mask, other=0).to(tl.int32)
    lo = tl.load(lut_ptr + (packed & 0xF))
    hi = tl.load(lut_ptr + ((packed >> 4) & 0xF))

    scale_base = scales_ptr.to(tl.int64)
    scale_idx = row * (PACKED_D // 8) + offsets // 8
    if e4m3_native_cx():
        scale_base = scale_base.to(tl.pointer_type(tl.float8e4nv))
        scale = tl.load(scale_base + scale_idx, mask=mask, other=0.0).to(tl.float32)
    else:
        scale_base = scale_base.to(tl.pointer_type(tl.uint8))
        scale = e4m3_u8_to_f32(tl.load(scale_base + scale_idx, mask=mask, other=0))
    scale *= tl.load(scale_2_ptr + global_row // rows_per_shard)
    scale = tl.where(in_range, scale, 0.0)
    out_base = out_row * (PACKED_D * 2)
    tl.store(out_ptr + out_base + offsets * 2, lo * scale, mask=mask)
    tl.store(out_ptr + out_base + offsets * 2 + 1, hi * scale, mask=mask)


def ple_nvfp4_rows(
    packed, scales, num_rows: int, embed_dim: int, row_ids: torch.Tensor,
    out: torch.Tensor, scale_2: torch.Tensor, rows_per_shard: int, *, gather: bool = True,
) -> torch.Tensor:
    """Gather and dequantize group-16 NVFP4 PLE rows; inputs may be pinned-host pointers."""
    n = row_ids.numel()
    assert embed_dim % 16 == 0
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        packed_d = embed_dim // 2
        _ple_nvfp4_kernel[(n,)](
            packed, scales, row_ids, out, _e2m1_lut(out.device.index), scale_2,
            num_rows, rows_per_shard,
            PACKED_D=packed_d, GATHER=gather, BLOCK_D=triton.next_power_of_2(packed_d),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_rows", "ple_nvfp4_rows"]
