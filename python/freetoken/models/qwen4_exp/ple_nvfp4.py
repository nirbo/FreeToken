"""Published group-16 NVFP4 PLE sidecar: discovery, RAM/UVA and direct-disk backends."""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
from freetoken.utils import init_logger

from .ple import PinnedUVATable
from .ple_disk import DiskRowTable, PleRowSource
from .weight import _safetensors_header

logger = init_logger(__name__)

_LAYOUT = "group16_e2m1_e4m3scale_lownibblefirst"
_SHARD_RE = re.compile(r"shard_(\d+)\.safetensors$")


@dataclass(frozen=True)
class Nvfp4PleSource:
    folder: str
    packed: PleRowSource
    scales: PleRowSource
    scale_2: Tuple[float, ...]

    @property
    def nbytes(self) -> int:
        return self.packed.total_rows * (self.packed.row_bytes + self.scales.row_bytes)


@dataclass
class Nvfp4PinnedTable:
    bank: object
    packed: torch.Tensor
    scales: torch.Tensor
    scale_2: Tuple[float, ...]


def _parse_source(folder: str) -> Nvfp4PleSource:
    with open(os.path.join(folder, "META.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("layout") != _LAYOUT:
        raise ValueError(f"unsupported layout {meta.get('layout')!r}")
    shards, total_rows, width = (int(meta[k]) for k in ("shards", "rows", "width"))
    if width % 16 or total_rows % shards:
        raise ValueError(f"invalid geometry: {total_rows} rows, width {width}, {shards} shards")

    indexed = {}
    for name in os.listdir(folder):
        if match := _SHARD_RE.fullmatch(name):
            indexed[int(match.group(1))] = os.path.join(folder, name)
    if sorted(indexed) != list(range(shards)):
        raise ValueError(f"needs shards 0..{shards - 1}, found {sorted(indexed)[:8]}")

    rows = total_rows // shards
    paths = [indexed[i] for i in range(shards)]
    packed_bases, scale_bases, scale_2 = [], [], []
    for path in paths:
        header, base = _safetensors_header(path)
        packed, scales, global_scale = (
            header.get("weight_e2m1"), header.get("weight_scale"), header.get("weight_scale_2")
        )
        if not packed or packed.get("dtype") != "U8" or packed.get("shape") != [rows, width // 2]:
            raise ValueError(f"bad weight_e2m1 tensor in {path}")
        if not scales or scales.get("dtype") != "F8_E4M3" or scales.get("shape") != [rows, width // 16]:
            raise ValueError(f"bad weight_scale tensor in {path}")
        if not global_scale or global_scale.get("dtype") != "F32" or global_scale.get("shape") != []:
            raise ValueError(f"bad weight_scale_2 tensor in {path}")
        packed_bases.append(base + packed["data_offsets"][0])
        scale_bases.append(base + scales["data_offsets"][0])
        with open(path, "rb") as fh:
            fh.seek(base + global_scale["data_offsets"][0])
            scale_2.append(struct.unpack("<f", fh.read(4))[0])

    common = dict(paths=paths, extent_file=list(range(shards)), rows_per_extent=rows)
    return Nvfp4PleSource(
        folder,
        PleRowSource(extent_base=packed_bases, row_bytes=width // 2, row_stride=width // 2,
                     scale=1.0, **common),
        PleRowSource(extent_base=scale_bases, row_bytes=width // 16, row_stride=width // 16,
                     scale=1.0, **common),
        tuple(scale_2),
    )


def resolve_nvfp4_source(
    model_folder: str, quant_path: str | None = None, *,
    expected_rows: int | None = None, expected_width: int | None = None,
) -> Nvfp4PleSource | None:
    """Find ``ples_nvfp4`` inside/beside the model; an explicit path fails loudly."""
    if quant_path:
        root = os.path.abspath(os.path.expanduser(quant_path))
        candidates = [root if os.path.isfile(os.path.join(root, "META.json"))
                      else os.path.join(root, "ples_nvfp4")]
    else:
        candidates = [os.path.join(model_folder, "ples_nvfp4")]
        parent = os.path.dirname(os.path.abspath(model_folder))
        try:
            candidates += [os.path.join(parent, name, "ples_nvfp4") for name in sorted(os.listdir(parent))]
        except OSError:
            pass

    errors = []
    for folder in dict.fromkeys(candidates):
        if not os.path.isfile(os.path.join(folder, "META.json")):
            continue
        try:
            source = _parse_source(folder)
            if expected_rows is not None and source.packed.total_rows < expected_rows:
                raise ValueError(f"has {source.packed.total_rows} rows, needs at least {expected_rows}")
            if expected_width is not None and source.packed.row_bytes * 2 != expected_width:
                raise ValueError(f"has width {source.packed.row_bytes * 2}, expected {expected_width}")
            return source
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{folder}: {exc}")
    if quant_path:
        raise ValueError("invalid --ple-quant-path: " + (errors[0] if errors else candidates[0]))
    if errors:
        logger.warning_rank0("ignoring incompatible NVFP4 PLE sidecar: " + errors[0])
    return None


def host_memory_info() -> tuple[int, int] | None:
    """Linux ``(MemTotal, MemAvailable)`` bytes, or None when unavailable."""
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, value, *_ = line.split()
                if key in ("MemTotal:", "MemAvailable:"):
                    values[key] = int(value) * 1024
    except OSError:
        return None
    return (values["MemTotal:"], values["MemAvailable:"]) if len(values) == 2 else None


def prefer_pinned_ple(ple_bytes: int, expert_bytes: int, memory: tuple[int, int] | None) -> bool:
    """Fit PLE + future experts while preserving 10% RAM or 12 GiB, whichever is larger."""
    if memory is None:
        return False
    total, available = memory
    return available >= ple_bytes + expert_bytes + max(12 << 30, total // 10)


def load_nvfp4_table(source: Nvfp4PleSource, *, workers: int = 8,
                     chunk: int = 8 << 20) -> Nvfp4PinnedTable:
    """Load weights and scales into one bank, then register it atomically for UVA."""
    from freetoken.moe.host_banks import HostBank, read_range_into
    from freetoken.utils.progress import byte_bar

    rows = source.packed.total_rows
    packed_bytes = rows * source.packed.row_bytes
    bank = HostBank((source.nbytes,), torch.uint8)
    bar = byte_bar(source.nbytes, "Loading NVFP4 PLE table")
    try:
        buf = bank.memoryview()
        for i, path in enumerate(source.packed.paths):
            nbytes = source.packed.rows_per_extent * source.packed.row_bytes
            read_range_into(buf, path, file_offset=source.packed.extent_base[i], nbytes=nbytes,
                            dest_offset=i * nbytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
        for i, path in enumerate(source.scales.paths):
            nbytes = source.scales.rows_per_extent * source.scales.row_bytes
            read_range_into(buf, path, file_offset=source.scales.extent_base[i], nbytes=nbytes,
                            dest_offset=packed_bytes + i * nbytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
        if torch.cuda.is_available():
            bank.pin()
    except BaseException:
        bank.release()
        raise
    finally:
        bar.close()
    packed = bank.tensor[:packed_bytes].view(rows, source.packed.row_bytes)
    scales = bank.tensor[packed_bytes:].view(torch.float8_e4m3fn).view(rows, source.scales.row_bytes)
    return Nvfp4PinnedTable(bank, packed, scales, source.scale_2)


class PinnedNVFP4Table(PinnedUVATable):
    """NVFP4 PLE in pinned host RAM, gathered and dequantized over UVA."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, scale_2: Sequence[float], *,
                 device: torch.device | None = None, prefetch: bool = True) -> None:
        assert packed.device.type == scales.device.type == "cpu"
        assert packed.dtype == torch.uint8 and scales.dtype == torch.float8_e4m3fn
        assert packed.is_contiguous() and scales.is_contiguous()
        self.weight, self.scales = packed, scales
        self.num_rows, packed_dim = packed.shape
        self.head_dim, self.dtype = packed_dim * 2, torch.bfloat16
        assert scales.shape == (self.num_rows, self.head_dim // 16)
        self._device = device or torch.device("cuda", torch.cuda.current_device())
        self.scale = torch.tensor(scale_2, dtype=torch.float32, device=self._device)
        assert self.num_rows % self.scale.numel() == 0
        self._rows_per_shard = self.num_rows // self.scale.numel()
        self._table_ptr, self._scale_ptr = device_ptr(packed), device_ptr(scales)
        self._stream = torch.cuda.Stream(device=self._device) if prefetch else None
        self._staging: torch.Tensor | None = None
        self._graph_staging: dict[int, torch.Tensor] = {}
        self._pending: Tuple[torch.Tensor, torch.Tensor] | None = None

    def _gather(self, row_ids: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_nvfp4_rows
        return ple_nvfp4_rows(self._table_ptr, self._scale_ptr, self.num_rows, self.head_dim,
                              row_ids.reshape(-1), dst, self.scale, self._rows_per_shard)


class Nvfp4DiskRowTable(DiskRowTable):
    """Two direct-I/O row stores feeding the same GPU NVFP4 dequant kernel."""

    def __init__(self, source: Nvfp4PleSource, hash_constants: dict, **kwargs) -> None:
        from freetoken.kernel import _ple_store

        super().__init__(source.packed, hash_constants, **kwargs)
        self._packed_dim = source.packed.row_bytes
        self.head_dim = self._packed_dim * 2
        self.scale = torch.tensor(source.scale_2, dtype=torch.float32, device=self._device)
        self._rows_per_shard = source.packed.rows_per_extent
        self._scale_store = _ple_store.PleStore(
            paths=list(source.scales.paths), extent_file=list(source.scales.extent_file),
            extent_base=list(source.scales.extent_base), rows_per_extent=source.scales.rows_per_extent,
            row_bytes=source.scales.row_bytes, row_stride=source.scales.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=[int(x) for x in hash_constants["per_head_vocab_sizes"]],
            head_offsets=[int(x) for x in hash_constants["per_head_offsets"]],
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv("FREETOKEN_PLE_IO_URING", "1") != "0",
        )
        self._scale_token_bytes = self.heads * source.scales.row_bytes
        graph_rows = self._graph_pinned.numel() // self._token_bytes
        eager_rows = self._eager_pinned.numel() // self._token_bytes
        self._graph_scale_pinned = alloc_pinned_tensor(graph_rows * self._scale_token_bytes,
                                                        dtype=torch.uint8).zero_()
        self._graph_scale_dev = torch.empty_like(self._graph_scale_pinned, device=self._device)
        self._eager_scale_pinned = alloc_pinned_tensor(eager_rows * self._scale_token_bytes,
                                                        dtype=torch.uint8).zero_()
        self._eager_scale_dev = torch.empty_like(self._eager_scale_pinned, device=self._device)
        logger.info_rank0(f"PLE NVFP4 sidecar: {source.nbytes / 2**30:.2f} GiB, {source.folder}")

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        packed = self._graph_pinned if graph else self._eager_pinned
        scales = self._graph_scale_pinned if graph else self._eager_scale_pinned
        offset = 0
        for run in runs:
            tokens = run.numel() - 2
            self._store.stage(run.data_ptr(), tokens, packed.data_ptr() + offset * self._token_bytes)
            self._scale_store.stage(run.data_ptr(), tokens,
                                    scales.data_ptr() + offset * self._scale_token_bytes)
            offset += tokens
        self._store.flush(0)
        self._scale_store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel import _ple_store
            _ple_store.memop_wait_reset(torch.cuda.current_stream(self._device).cuda_stream,
                                        self._flag.data_ptr())
        packed, packed_dev = ((self._graph_pinned, self._graph_dev) if capturing
                              else (self._eager_pinned, self._eager_dev))
        scales, scale_dev = ((self._graph_scale_pinned, self._graph_scale_dev) if capturing
                             else (self._eager_scale_pinned, self._eager_scale_dev))
        rows = row_ids.shape[0]
        packed_n, scale_n = rows * self._token_bytes, rows * self._scale_token_bytes
        packed_dev[:packed_n].copy_(packed[:packed_n], non_blocking=True)
        scale_dev[:scale_n].copy_(scales[:scale_n], non_blocking=True)

        from freetoken.kernel.triton.ple import ple_nvfp4_rows
        flat_rows = row_ids.numel()
        dst = out.view(flat_rows, self.head_dim) if out is not None else torch.empty(
            (flat_rows, self.head_dim), dtype=self.dtype, device=self._device)
        ple_nvfp4_rows(packed_dev, scale_dev, self.num_rows, self.head_dim,
                       row_ids.reshape(-1), dst, self.scale, self._rows_per_shard, gather=False)
        return out if out is not None else dst.view(*row_ids.shape[:-1], -1)
