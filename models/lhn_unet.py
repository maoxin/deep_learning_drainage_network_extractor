from argparse import ArgumentParser

import torch
from pytorch_lightning.core.lightning import LightningModule

from models.unet import DoubleConv, Down, Up, OutConv


class LHNUNet(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_classes_l1', type=int, default=8)
        parser.add_argument('--n_classes_l2', type=int, default=8)
        parser.add_argument('--n_classes_l3', type=int, default=8)
        parser.add_argument('--n_classes_l4', type=int, default=8)
        parser.fromfile_prefix_chars = "@"

        return parser

    def __init__(self, n_channels, n_classes,
                 n_classes_l1, n_classes_l2, n_classes_l3, n_classes_l4):
        super(LHNUNet, self).__init__()
        self.save_hyperparameters()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.outc_l4 = OutConv(1024 // factor, n_classes_l4)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.outc_l3 = OutConv(512 // factor, n_classes_l3)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.outc_l2 = OutConv(256 // factor, n_classes_l2)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.outc_l1 = OutConv(128 // factor, n_classes_l1)
        self.up4 = Up(128, 64, self.bilinear)
        # self.dp = nn.Dropout2d(64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_l4 = self.outc_l4(x5)
        x = self.up1(x5, x4)
        x_l3 = self.outc_l3(x)
        x = self.up2(x, x3)
        x_l2 = self.outc_l2(x)
        x = self.up3(x, x2)
        x_l1 = self.outc_l1(x)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x, x_l1, x_l2, x_l3, x_l4


if __name__ == "__main__":
    x = torch.randn(2, 1, 512, 256)

    unet = LHNUNet(n_channels=1, n_classes=2, n_classes_l1=4, n_classes_l2=8,
                   n_classes_l3=10, n_classes_l4=12)
    y, y_l1, y_l2, y_l3, y_l4 = unet(x)
