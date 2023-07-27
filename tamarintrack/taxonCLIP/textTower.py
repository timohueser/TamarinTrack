""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from ..config import TextTowerConfig
from .utils import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


# text_cfg.hf_model_name,
# output_dim=embed_dim,
# proj=text_cfg.proj,
# pooler_type=text_cfg.pooler_type,
# pretrained=text_cfg.hf_model_pretrained,
# output_tokens=text_cfg.output_tokens,


class TextTower(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        cfg: TextTowerConfig,
        # model_name_or_path: str,
        # output_dim: int,
        # config: PretrainedConfig = None,
        # pooler_type: str = None,
        # proj: str = None,
        # pretrained: bool = True,
        # output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = cfg.output_tokens
        self.output_dim = cfg.embedding_dim

        self.config = AutoConfig.from_pretrained(cfg.hf_model_name)
        create_func, model_args = (
            (AutoModel.from_pretrained, cfg.hf_model_name)
            if cfg.pretrained
            else (AutoModel.from_config, self.config)
        )
        # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
        if (
            hasattr(self.config, "is_encoder_decoder")
            and self.config.is_encoder_decoder
        ):
            self.transformer = create_func(model_args)
            self.transformer = self.transformer.encoder
        else:
            self.transformer = create_func(
                model_args
            )  # , add_pooling_layer=uses_transformer_pooler)
        if cfg.pooler_type is None:  # get default arch pooler
            cfg.pooler_type = arch_dict[self.config.model_type]["pooler"]

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, "vocab_size", 0)
        self.context_length = getattr(self.config, "max_position_embeddings", 0)

        self.pooler = _POOLERS[cfg.pooler_type]()

        d_model = getattr(
            self.config, arch_dict[self.config.model_type]["config_names"]["width"]
        )
        if (d_model == self.output_dim) and (
            cfg.proj is None
        ):  # do we always need a proj?
            self.proj = nn.Identity()
        elif cfg.proj == "linear":
            self.proj = nn.Linear(d_model, self.output_dim, bias=False)
        elif cfg.proj == "mlp":
            hidden_size = (d_model + self.output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, self.output_dim, bias=False),
            )

    def forward(self, x: TensorType):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seq_len) != self.pooler.cls_token_position, :
            ]
            if type(self.pooler) == ClsPooler
            else out.last_hidden_state
        )

        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )
            return

        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, "encoder")
            else self.transformer
        )
        layer_list = getattr(
            encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"]
        )
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer,
            arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
                )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
