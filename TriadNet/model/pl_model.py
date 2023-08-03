from pytorch_lightning.core.lightning import LightningModule
import torch

import numpy as np

from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

from torch.optim import Adam
from TriadNet.dataloader.mri_datamodule import concatenate_channels
from TriadNet.model.triade_net import TriadeNet
from TriadNet.model.triade_loss import TriadeLoss

"""
Define a simple Pytorch Lightning model to train the TriadeNet model
"""


class PLModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.define_network()
        self.define_losses()

        self.metrics = [DiceMetric(include_background=False)]

    def define_network(self):
        print(f'Initializing 3D TriadeNet with n_classes = {self.hparams.n_classes}')

        self.net = TriadeNet(dim=3,
                             in_channels=self.hparams.in_channels,
                             out_channels=self.hparams.n_classes,
                             )


    def define_losses(self):
        self.loss = TriadeLoss(include_background=True)

    def configure_optimizers(self):
        """
        Define Optimizer
        :return:
        """
        print(f'Initializing Adam optimized with LR={self.hparams.learning_rate}')
        optimizer = Adam(self.net.parameters(), lr=self.hparams.learning_rate)

        return [optimizer]

    def logits_to_segm(self, logits, keep_dim=False):
        """
        Convert logits to 3d segmentation
        :param logits:
        :param keep_dim:
        :return:
        """
        prob = torch.softmax(logits, 1)
        seg = torch.argmax(prob, 1, keepdim=keep_dim)
        return seg

    def compute_loss(self,
                     pred_dict,
                     y: torch.Tensor,
                     step: str = 'train'):

        current_loss = self.loss.forward(pred_dict, y)
        self.log(step + '_loss', current_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return current_loss

    def forward(self, x: torch.Tensor):
        pred_dict = self.net(x)
        return pred_dict

    def metric_step(self, seg, y, step='train'):
        metric_dict = []
        batch_size = y.shape[0]

        for m in self.metrics:
            if isinstance(m, DiceMetric):
                for b in range(batch_size):
                    segb = seg[b][None, ...]
                    yb = y[b][None, ...]
                    segb_oh = one_hot(segb, self.hparams.n_classes)
                    yb_oh = one_hot(yb, self.hparams.n_classes)
                    m(segb_oh, yb_oh)
                    metric_dict += [{f'{step}_dice': m.aggregate()}]
                    m.reset()

        return metric_dict

    def training_step(self, batch, batch_idx):
        images_dict = concatenate_channels(patient_dict=batch)

        x = images_dict['image']
        y = images_dict['segm']

        pred_dict = self(x)
        loss = self.compute_loss(pred_dict, y, step='train')

        predicted_seg = self.logits_to_segm(pred_dict['logits']).unsqueeze(1)
        train_metrics = self.metric_step(predicted_seg, y, step='train')

        return {"loss": loss, 'train_metrics': train_metrics}

    def validation_step(self, batch, batch_idx):
        images_dict = concatenate_channels(patient_dict=batch)

        x = images_dict['image']
        y = images_dict['segm']

        pred_dict = self(x)
        loss = self.compute_loss(pred_dict, y, step='val')

        predicted_seg = self.logits_to_segm(pred_dict['logits']).unsqueeze(1)
        val_metrics = self.metric_step(predicted_seg, y, step='val')

        return {'val_loss': loss, 'val_metrics': val_metrics}

    def training_epoch_end(self, outputs):
        train_epoch_end_dict = {}
        all_train_metrics = []

        train_epoch_end_dict['train_loss'] = np.mean([x['loss'].item() for x in outputs])

        for x in outputs:
            all_train_metrics += x['train_metrics']

        all_keys = set().union(*all_train_metrics)
        for metric in all_keys:
            stacked_train_metrics = [x[metric].item() for x in all_train_metrics if metric in x]
            train_epoch_end_dict[metric] = np.mean(stacked_train_metrics)

        train_loss = train_epoch_end_dict.pop('train_loss')
        self.log_dict(train_epoch_end_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('mean_train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_epoch_end(self, val_step_outputs):
        """
        Function called at the end of validation : gather all validation metrics and plot to tensorboard.
        :param val_step_outputs:
        :return:
        """

        val_epoch_end_dict = {}
        all_val_metrics = []

        val_epoch_end_dict['val_loss'] = np.mean([x['val_loss'].item() for x in val_step_outputs])

        for x in val_step_outputs:
            all_val_metrics += x['val_metrics']

        all_keys = set().union(*all_val_metrics)
        for metric in all_keys:
            stacked_val_metrics = [x[metric].item() for x in all_val_metrics if metric in x]
            val_epoch_end_dict[metric] = np.mean(stacked_val_metrics)

        val_loss = val_epoch_end_dict.pop('val_loss')
        self.log_dict(val_epoch_end_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('mean_val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)





