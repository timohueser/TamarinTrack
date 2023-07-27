""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from timm.layers import AttentionPool2d as AbsAttentionPool2d
from timm.layers import RotAttentionPool2d
from timm.models.layers import Mlp, to_2tuple

from ..config import VisionTowerConfig
from .utils import freeze_batch_norm_2d


class VisionTower(nn.Module):
    def __init__(
        self,
        cfg: VisionTowerConfig,
        image_size: int = 224,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)

        patch_drop = cfg.patch_dropout if cfg.patch_dropout > 0 else None
        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if cfg.timm_drop_path is not None:
            timm_kwargs["drop_path_rate"] = cfg.timm_drop_path
        if patch_drop is not None:
            timm_kwargs["patch_drop_rate"] = patch_drop

        custom_pool = cfg.timm_pool in ("abs_attn", "rot_attn")
        if not cfg.timm_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            self.trunk = timm.create_model(
                cfg.timm_model_name,
                num_classes=cfg.embedding_dim,
                global_pool=cfg.timm_pool,
                pretrained=cfg.pretrained,
                **timm_kwargs,
            )
            prev_chs = cfg.embedding_dim
        else:
            self.trunk = timm.create_model(
                cfg.timm_model_name,
                pretrained=cfg.pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get("pool_size", None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool="")
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = {"global_pool": cfg.timm_pool} if cfg.timm_pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if cfg.timm_pool == "abs_attn":
            head_layers["pool"] = AbsAttentionPool2d(
                prev_chs, feat_size=feat_size, out_features=cfg.embedding_dim
            )
            prev_chs = cfg.embedding_dim
        elif cfg.timm_pool == "rot_attn":
            head_layers["pool"] = RotAttentionPool2d(
                prev_chs, out_features=cfg.embedding_dim
            )
            prev_chs = cfg.embedding_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if cfg.timm_proj == "linear":
            head_layers["drop"] = nn.Dropout(cfg.timm_drop)
            head_layers["proj"] = nn.Linear(
                prev_chs, cfg.embedding_dim, bias=cfg.timm_proj_bias
            )
        elif cfg.timm_proj == "mlp":
            head_layers["mlp"] = Mlp(
                prev_chs,
                2 * cfg.embedding_dim,
                cfg.embedding_dim,
                drop=(cfg.timm_drop, 0),
                bias=(True, cfg.timm_proj_bias),
            )
        else:
            assert not cfg.timm_proj, f"Unknown projection type {cfg.timm_proj}."

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_modules, group_parameters
            except ImportError as err:
                raise RuntimeError(
                    "Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`"
                ) from err
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception:
            logging.warning(
                "grad checkpointing not supported for this timm image tower, continuing without..."
            )

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
