from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match=None, name=""):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    "roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings",
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/distilbert#overview
    "distilbert": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings",
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    "xlm-roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings",
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    "mt5": {
        "config_names": {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            "context_length": "",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "num_heads",
            "layers": "num_layers",
            "layer_attr": "block",
            "token_embeddings_attr": "embed_tokens",
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    "bert": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
        },
        "pooler": "cls_pooler",
    },
}
