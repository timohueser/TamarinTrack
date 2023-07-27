import torch
from torch import nn


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_multi_label_ground_truth(self, device, names, valid_names):
        labels = torch.zeros((len(names), len(names)), device=device)
        valid_names_sets = [set(valid_name_list) for valid_name_list in valid_names]
        for i, name in enumerate(names):
            for j, valid_name_set in enumerate(valid_names_sets):
                if name in valid_name_set:
                    labels[i, j] = 1
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(
        self, image_features, text_features, logit_scale, names=None, valid_names=None
    ):
        valid_names = [list(row) for row in zip(*valid_names, strict=True)]

        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        # Get binary ground truth labels for each class
        labels = self.get_multi_label_ground_truth(device, names, valid_names)
        pos_weight = (
            torch.ones([labels.size(1)]).cuda() * 20
        )  # TODO: this 20 is super arbitrary!
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        total_loss = (
            loss_func(logits_per_image, labels.T) + loss_func(logits_per_text, labels)
        ) / 2

        return total_loss
