"""Qwen3.8-Flash-Next multi-token-prediction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMTPMoE

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.layers import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


class Qwen4ExpMTPDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMTPMoE(config)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        hidden = self.attn_hyper_connection.combine(
            hidden, self.self_attn.forward(block_input, batch), inject
        )
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTP(BaseOP):
    """One reusable MTP layer; the target embedding and LM head remain shared."""

    def __init__(self, config: ModelConfig, embed_tokens: VocabParallelEmbedding) -> None:
        args = config.qwen4_args
        if args.mtp_num_hidden_layers != 1:
            raise ValueError(
                f"Qwen3.8 MTP needs exactly one checkpoint layer, got {args.mtp_num_hidden_layers}"
            )
        self._embed_tokens = embed_tokens  # shared, intentionally absent from this state dict
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            config.hidden_size, config.rms_norm_eps, 1
        )
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, args.hc_count
        )
        self.fc_embedding = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.fc_hidden = LinearReplicated(
            config.hidden_size, config.hidden_size, has_bias=False
        )
        self.layers = OPList([Qwen4ExpMTPDecoderLayer(config, config.num_layers)])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def forward(
        self, input_ids: torch.Tensor, previous_hidden: torch.Tensor, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.fc_embedding.forward(
            self.pre_fc_norm_embedding.forward(self._embed_tokens.forward(input_ids))
        )
        hidden = self.pre_fc_norm_hidden.forward(previous_hidden)
        hidden = self.fc_hidden.forward(
            hidden.view(-1, self.hc_count, self.hidden_size)
        ) + embedding.unsqueeze(1)
        hidden = hidden.flatten(1)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        return self.hyper_connection_mixer.mix(hidden)[0], hidden


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMTPDecoderLayer"]
