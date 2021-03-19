from argparse import ArgumentParser
import math
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import pytorch_lightning.metrics.functional as mf


class DoubleConv(LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(LightningModule):
    """Double conv then downscaling with maxpool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutFC(LightningModule):
    def __init__(self, in_channels, out_channels):
        super(OutFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class SlideNet(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in-channels', type=int, default=12)
        parser.add_argument('--learning-rate', type=float, default=0.02)
        parser.add_argument('--adam-epsilon', type=float, default=1e-8)
        parser.fromfile_prefix_chars = "@"

        return parser

    def __init__(self, in_channels, n_classes=3,
                 learning_rate=0.02, adam_epsilon=1e-8, **kwargs):
        super(SlideNet, self).__init__()
        self.save_hyperparameters()

        self.n_channels = in_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fc = OutFC(512, n_classes)

    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return: {
            x_swnet: (B, 3),
        }
        """

        x = self.forward_core(x)

        return {
            'x_swnet': x,
        }


    def forward_core(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.avg_pooling(x).flatten(1)
        x = self.out_fc(x)

        return x

    def configure_optimizers(self):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
        )

        steps_per_epoch = math.ceil(12372583 * 1 * 1 / (self.hparams.gpus * self.hparams.batch_size))
        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                    epochs=self.hparams.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x_swnet = self(batch['x'])['x_swnet'] # (B, 3)
        y_swnet = batch['y_swnet'] # (B,)
        loss_swnet = F.cross_entropy(x_swnet, y_swnet) # (B, 3)

        acc_swnet, kc_swnet = self.metrics(x_swnet.max(dim=1)[1], y_swnet)

        logs = {
            'train/loss': loss_swnet,
            'train/acc_swnet': acc_swnet,
            'train/kc_swnet': kc_swnet,
        }

        return {'loss': loss_swnet, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x_swnet = self(batch['x'])['x_swnet']  # (B, 3)
        y_swnet = batch['y_swnet']  # (B,)

        loss_swnet = F.cross_entropy(x_swnet, y_swnet)
        acc_swnet, kc_swnet = self.metrics(x_swnet.max(dim=1)[1], y_swnet)

        return {'val_loss': loss_swnet,
                'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {}
        for item in outputs:
            for key in item:
                if key not in logs:
                    logs[key] = []
                if torch.isnan(item[key]).any():
                    continue
                logs[key].append(item[key])
        logs = {'val/'+key: torch.stack(logs[key]).mean() for key in logs}

        return {
            'val_loss': avg_loss,
            'log': logs,
        }

    def test_step(self, batch, batch_idx):
        x_swnet = self(batch['x'])['x_swnet']  # (B, 3)
        y_swnet = batch['y_swnet']  # (B,)



        loss_swnet = F.cross_entropy(x_swnet, y_swnet)
        acc_swnet, kc_swnet, f1_swnet, precision_swnet, recall_swnet = self.metrics(x_swnet.max(dim=1)[1], y_swnet,
                                                                                    is_swnet=True)

        return {'test_loss': loss_swnet,
                'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet,
                'f1_swnet': f1_swnet,
                'precision_swnet': precision_swnet, 'recall_swnet': recall_swnet,
                }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {}
        for item in outputs:
            for key in item:
                if key not in logs:
                    logs[key] = []
                logs[key].append(item[key])
        new_logs = {}
        for key in logs:
            if 'precision' not in key and 'recall' not in key and 'f1' not in key:
                val = torch.stack(logs[key])
                val[torch.isnan(val)] = 1
                new_logs['test/' + key] = val.mean()
            else:
                new_logs = {**new_logs, **self.unpack_multi_class('test/' + key, torch.stack(logs[key]).mean(0))}
        # logs = {'test/'+key: torch.stack(logs[key]).mean() for key in logs}

        return {
            'test_loss': avg_loss,
            'log': new_logs,
        }

    def unpack_multi_class(self, key, val):
        result = {}
        for i, v in enumerate(val):
            result[f"{key}_{i}"] = v

        return result

    def infer(self, batch):
        arg_to_device_list = [
            'x'
        ]

        for arg in arg_to_device_list:
            try:
                batch[arg] = batch[arg].to(self.device)
            except KeyError:
                pass

        x = batch['x'].to(self.device)
        y_swnet = batch['y_swnet'] # (B,)
        x_swnet = self(batch['x'])['x_swnet'].argmax(1)  # (B,)

        return x_swnet.squeeze(0), y_swnet.squeeze(0)

    def metrics(self, pred, target, num_classes=3, is_swnet=False):
        if is_swnet:
            pred[pred == 2] = 1
            target[target == 2] = 1
            num_classes = 2

        confusion_m = mf.confusion_matrix(pred, target, num_classes=num_classes)

        # acc
        # try:
        accuracy = confusion_m.diag().sum() / len(pred)
        # except:
        #     print("pred:")
        #     print(pred)
        #     print("target:")
        #     print(target)
        #     print("confusion_m:")
        #     print("confusion_m")
        #     print("---")
        #     accuracy = 0

        # kappa
        # try:
        p0 = accuracy
        pc = 0
        for i in range(confusion_m.shape[0]):
            pc = pc + confusion_m[i].sum() * confusion_m[:, i].sum()
        pc = pc / len(pred)**2
        kc = (p0 - pc) / (1 - pc)
        # if pc != 1:
        #     kc = (p0 - pc) / (1 - pc)
        # else:
        #     kc = torch.tensor(1.0)
        #     kc.to(p0.device)
        # except:
        #     kc = 0

        f1 = mf.f1_score(pred, target, num_classes=num_classes, class_reduction='none')
        precision = mf.precision(pred, target, num_classes=num_classes, class_reduction='none')
        recall = mf.recall(pred, target, num_classes=num_classes, class_reduction='none')

        return accuracy, kc, f1, precision, recall

if __name__ == "__main__":
    x = torch.randn(2, 1, 512, 256)

    slide_net = SlideNet(in_channels=1, n_classes=2)
    y = slide_net(x)