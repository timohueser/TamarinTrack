import torch
import torch.nn.functional as F
from torch import nn


class ClipMetrics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(
        self, image_features, text_features, logit_scale, all_names, all_valid_names
    ):
        metrics = {}
        logits_per_image = (logit_scale * image_features @ text_features.t()).detach()
        logits_per_text = logits_per_image.t().detach()
        all_valid_names = [list(row) for row in zip(*all_valid_names, strict=True)]
        ground_truth = self.get_multi_label_ground_truth(
            all_names, all_valid_names, image_features
        )

        logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}

        for name, logit in logits.items():
            if name == "image_to_text":
                gt = ground_truth.t()
            else:
                gt = ground_truth
            logit = F.sigmoid(logit)
            logit[logit >= 0.5] = 1
            logit[logit < 0.5] = 0
            TP = torch.sum((logit == 1) & (gt == 1))
            FP = torch.sum((logit == 1) & (gt == 0))
            torch.sum((logit == 0) & (gt == 0))
            FN = torch.sum((logit == 0) & (gt == 1))

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            preds = (logit == gt).float()
            preds = preds.detach()
            metrics[f"{name}_acc"] = torch.mean(preds) * 100
            metrics[f"{name}_precision"] = precision.item()
            metrics[f"{name}_recall"] = recall.item()

        return metrics

    def get_multi_label_ground_truth(self, names, valid_names, image_features):
        labels = torch.zeros((len(names), len(names))).type_as(image_features)
        valid_names_sets = [set(valid_name_list) for valid_name_list in valid_names]
        for i, name in enumerate(names):
            for j, valid_name_set in enumerate(valid_names_sets):
                if name in valid_name_set:
                    labels[i, j] = 1
        return labels
