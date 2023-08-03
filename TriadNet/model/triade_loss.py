import torch
from monai.losses import TverskyLoss


class TriadeLoss:
    """
    Segmentation loss to get predictive intervals by predicting mean mask, upper mask, lower mask
    """

    def __init__(self,
                 tol: int = 0.2,
                 batch: bool = True,
                 include_background: bool = True,
                 reduction='mean',
                 ):
        self.tol = tol
        self.include_background = include_background
        self.reduction = reduction
        self.loss_mean = TverskyLoss(include_background=include_background,
                                     to_onehot_y=True,
                                     sigmoid=True,
                                     alpha=0.5, beta=0.5,
                                     batch=batch)

        self.loss_upper = TverskyLoss(include_background=include_background,
                                      to_onehot_y=True,
                                      sigmoid=True,
                                      alpha=self.tol, beta=1-self.tol,  # low penalty on FP, high on FN
                                      batch=batch)

        self.loss_lower = TverskyLoss(include_background=include_background,
                                      to_onehot_y=True,
                                      sigmoid=True,
                                      alpha=1-self.tol, beta=self.tol,  # high penalty on FP, low on FN
                                      batch=batch)

    def forward(self, predictive_dict, target: torch.Tensor):

        upper = predictive_dict['upper']
        lower = predictive_dict['lower']
        mean = predictive_dict['logits']

        base_loss = self.loss_mean(mean, target)
        low_loss = self.loss_lower(lower, target)
        high_loss = self.loss_upper(upper, target)

        tot_loss = (base_loss + low_loss + high_loss) / 3

        return tot_loss


