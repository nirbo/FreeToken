import torch

import freetoken.layers.rotary as rotary
from freetoken.models.qwen4_exp.config import enable_mtp
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTP
from freetoken.utils.torch_utils import torch_dtype

from .common import Fixture, parsed_config, requires_cuda


def _meta_model(config):
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"):
            return Qwen4ExpForCausalLM(config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def test_mtp_model_matches_the_loader_state_layout():
    config = enable_mtp(parsed_config(mtp_num_hidden_layers=1), 3)
    model = _meta_model(config)

    keys = {name for name in model.state_dict() if name.startswith("mtp.")}
    assert len(keys) == 34
    assert {
        "mtp.layers.0.mlp.experts.gate_up_packed",
        "mtp.layers.0.mlp.experts.gate_up_scale",
        "mtp.layers.0.mlp.experts.gate_up_global",
        "mtp.layers.0.mlp.experts.down_packed",
        "mtp.layers.0.mlp.experts.down_scale",
        "mtp.layers.0.mlp.experts.down_global",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight_scale",
        "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight_global",
        "mtp.layers.0.self_attn.qkv_proj.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
    } <= keys


def test_mtp_stays_absent_when_disabled():
    config = parsed_config(mtp_num_hidden_layers=1)
    model = _meta_model(config)
    assert model.mtp is None
    assert not any(name.startswith("mtp.") for name in model.state_dict())


def test_mtp_extends_qsa_cache_geometry():
    from freetoken.kvcache import create_kvcache_pool

    config = enable_mtp(parsed_config(mtp_num_hidden_layers=1), 3)
    full = next(group for group in config.attention_groups if group.name == "full")
    assert full.layer_ids[-1] == config.num_layers
    assert full.num_index_layers == len(full.layer_ids)

    pool = create_kvcache_pool(
        model_config=config,
        num_pages=2,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_req_slots=2,
    )
    assert pool.ring_capacity == 8
    assert pool.k_cache(config.num_layers).shape[0] == 2


@requires_cuda
def test_mtp_forward_runs_on_the_virtual_qsa_layer():
    from freetoken.layers import VocabParallelEmbedding

    config = enable_mtp(parsed_config(mtp_num_hidden_layers=1), 3)
    fixture = Fixture(config, num_pages=8, max_running_req=1)
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        mtp = Qwen4ExpMTP(config, embedding)
    embedding.weight.normal_(0, 0.02)
    for name, tensor in mtp.state_dict().items():
        if tensor.dtype == torch.uint8:
            tensor.zero_()
        elif tensor.dtype == torch.float8_e4m3fn or name.endswith("_global"):
            tensor.fill_(1)
        else:
            tensor.normal_(0, 0.02)

    req = fixture.req(table_idx=0, cached_len=0, device_len=3)
    batch = fixture.batch([req], "prefill")
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    previous = torch.randn(3, config.qwen4_args.ple_state_width, device="cuda", dtype=torch.bfloat16)
    with fixture.ctx.forward_batch(batch):
        output, hidden = mtp.forward(input_ids, previous, batch)
    assert output.shape == (3, config.hidden_size)
    assert hidden.shape == previous.shape
    assert torch.isfinite(output).all()
