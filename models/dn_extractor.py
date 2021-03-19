from argparse import ArgumentParser
import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning.metrics.functional as mf
import numpy as np

from models.unet import UNet
from models.lhn_unet import LHNUNet
from models.deep_lab_v3plus.deeplab import DeepLab
from models.modsegnet import ModSegNet
from models.segnet import SegNet
from models.aspp_segnet import ASPPSegNet
from models.sp_segnet import SPSegNet
from models.dl_segnet import DLSegNet


class DrainageNetworkExtractor(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--use-coord', action='store_true')
        parser.add_argument('--use-d8', action='store_true')
        parser.add_argument('--use-slope', action='store_true')
        parser.add_argument('--use-curvature', action='store_true')
        parser.add_argument('--in-channels', type=int, default=12)
        parser.add_argument('--use-model', type=str, default='unet')
        parser.add_argument('--learning-rate', type=float, default=0.02)
        parser.add_argument('--adam-epsilon', type=float, default=1e-8)
        parser.add_argument('--use-kriging-loss', action='store_true')
        parser.fromfile_prefix_chars = "@"

        temp_args, _ = parser.parse_known_args()
        if temp_args.use_model.lower() == 'unet':
            parser = UNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'lhn_unet':
            parser = LHNUNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'deep_lab':
            parser = DeepLab.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'modsegnet':
            parser = ModSegNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'segnet':
            parser = SegNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'aspp_segnet':
            parser = ASPPSegNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'sp_segnet':
            parser = SPSegNet.add_model_specific_args(parser)
        elif temp_args.use_model.lower() == 'dl_segnet':
            parser = DLSegNet.add_model_specific_args(parser)

        return parser

    def __init__(self, in_channels=12, use_model='unet', use_d8=False,
                 learning_rate=0.02, adam_epsilon=1e-8, **kwargs):
        super(DrainageNetworkExtractor, self).__init__()
        self.save_hyperparameters()

        if use_model.lower() == 'unet':
            self.model = UNet(n_channels=in_channels, n_classes=12, bilinear=self.hparams.bilinear)
        elif use_model.lower() == 'lhn_unet':
            self.model = LHNUNet(n_channels=in_channels, n_classes=12,
                                 n_classes_l1=self.hparams.n_classes_l1, n_classes_l2=self.hparams.n_classes_l2,
                                 n_classes_l3=self.hparams.n_classes_l3, n_classes_l4=self.hparams.n_classes_l4)
        elif use_model.lower() == 'deep_lab':
            self.model = DeepLab(backbone=self.hparams.backbone, in_channels=in_channels, num_classes=12,
                                 sync_bn=self.hparams.sync_bn, freeze_bn=self.hparams.freeze_bn,
                                 output_stride=self.hparams.output_stride)
        elif use_model.lower() == 'modsegnet':
            self.model = ModSegNet(num_classes=12, n_init_features=in_channels, drop_rate=self.hparams.drop_rate)
        elif use_model.lower() == 'segnet':
            self.model = SegNet(num_classes=12, n_init_features=in_channels, drop_rate=self.hparams.drop_rate,
                                use_kriging_loss=self.hparams.use_kriging_loss)
        elif use_model.lower() == 'aspp_segnet':
            self.model = ASPPSegNet(num_classes=12, n_init_features=in_channels,
                                    use_kriging_loss=self.hparams.use_kriging_loss)
        elif use_model.lower() == 'sp_segnet':
            self.model = SPSegNet(num_classes=12, n_init_features=in_channels)
        elif use_model.lower() == 'dl_segnet':
            self.model = DLSegNet(num_classes=12, n_init_features=in_channels,
                                  drop_rate=self.hparams.drop_rate)
        else:
            raise Exception(f"{use_model} is not implemented")

        if use_d8:
            self.d8_emb = nn.Embedding(9, 3, max_norm=1)


    def forward(self, x):
        """

        :param x: (B, C, H, W)
        :return: {
            x_fdr: (B, 9, H, W),
            x_swnet: (B, 3, H, W),
        }
        """
        if self.hparams.use_model.lower() != 'lhn_unet':
            x = self.model(x)
            if self.hparams.use_kriging_loss:
                x, loss_kriging = x
            # x_fdr = torch.softmax(x[:, :9], dim=1)
            # x_swnet = torch.softmax(x[:, 9:], dim=1)

            x_fdr = x[:, :9]
            x_swnet = x[:, 9:]

            if not self.hparams.use_kriging_loss:
                return {
                    'x_fdr': x_fdr,
                    'x_swnet': x_swnet,
                }
            else:
                return {
                    'x_fdr': x_fdr,
                    'x_swnet': x_swnet,
                    'loss_kriging': loss_kriging,
                }
        else:
            x, x_l1, x_l2, x_l3, x_l4 = self.model(x)
            # x_fdr = torch.softmax(x[:, :9], dim=1)
            # x_swnet = torch.softmax(x[:, 9:], dim=1)

            x_fdr = x[:, :9]
            x_swnet = x[:, 9:]

            return {
                'x_fdr': x_fdr,
                'x_swnet': x_swnet,
                'x_l1': x_l1, 'x_l2': x_l2, 'x_l3': x_l3, 'x_l4': x_l4,
            }

    def configure_optimizers(self):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
        )
        # scheduler = {
        #     'scheduler': ReduceLROnPlateau(optimizer),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        # steps_per_epoch = math.ceil(5217 * 8 * 3 / (self.hparams.gpus * self.hparams.batch_size))
        # steps_per_epoch = math.ceil(10990 * 6 * 3 / (self.hparams.gpus * self.hparams.batch_size))
        steps_per_epoch = math.ceil(197822 / (self.hparams.gpus * self.hparams.batch_size))
        scheduler = {
            # 'scheduler': OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
            #                         epochs=self.hparams.max_epochs, steps_per_epoch=6956),
            # 'scheduler': OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
            #                         epochs=self.hparams.max_epochs, steps_per_epoch=8348),
            # 'scheduler': OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
            #                         epochs=self.hparams.max_epochs, steps_per_epoch=5217),
            'scheduler': OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                    epochs=self.hparams.max_epochs, steps_per_epoch=steps_per_epoch),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def process4tvt(self, batch, batch_idx):
        """
        process the results from forward() for train, val, and test steps
        :param batch:
        :param batch_idx:
        :return:
        """

        x = batch['x']
        if self.hparams.use_slope:
            x = torch.cat([x, batch['slope']], dim=1)

        if self.hparams.use_curvature:
            x = torch.cat([x, batch['curvature']], dim=1)

        if self.hparams.use_slope and self.hparams.use_curvature:
            x = torch.cat([x, batch['slope']**2 * batch['curvature']], dim=1)

        if self.hparams.use_d8:
            d8 = self.d8_emb(batch['d8'].squeeze(1)).permute(0, 3, 1, 2)
            x = torch.cat([x, d8], dim=1)


        mask = batch['mask']
        y = batch['y']

        result = self(x)

        x_fdr = result['x_fdr'].permute(0, 2, 3, 1)
        # (B, H, W, 9)
        x_swnet = result['x_swnet'].permute(0, 2, 3, 1)
        # (B, H, W, 3)

        y_fdr = y[:, 0]
        y_swnet = y[:, 1]
        # (B, H, W)

        # use mask
        mask_inds = mask == 255
        x_fdr_masked = x_fdr[mask_inds]
        # (L, 9)
        x_swnet_masked = x_swnet[mask_inds]
        # (L, 3)
        y_fdr_masked = y_fdr[mask_inds]
        # (L, )
        y_swnet_masked = y_swnet[mask_inds]
        # (L, )

        if self.hparams.use_model.lower() != 'lhn_unet' or not self.training:
            if not self.hparams.use_kriging_loss:
                return x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked
            else:
                return x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked, result['loss_kriging']
        else:
            mask_l1 = batch['mask_l1']
            mask_l2 = batch['mask_l2']
            mask_l3 = batch['mask_l3']
            mask_l4 = batch['mask_l4']
            y_l1 = batch['y_l1'].permute(0, 2, 3, 1)
            y_l2 = batch['y_l2'].permute(0, 2, 3, 1)
            y_l3 = batch['y_l3'].permute(0, 2, 3, 1)
            y_l4 = batch['y_l4'].permute(0, 2, 3, 1)
            x_l1 = result['x_l1'].permute(0, 2, 3, 1)
            x_l2 = result['x_l2'].permute(0, 2, 3, 1)
            x_l3 = result['x_l3'].permute(0, 2, 3, 1)
            x_l4 = result['x_l4'].permute(0, 2, 3, 1)

            mask_inds_l1 = mask_l1 == 1
            mask_inds_l2 = mask_l2 == 1
            mask_inds_l3 = mask_l3 == 1
            mask_inds_l4 = mask_l4 == 1
            x_l1_masked = x_l1[mask_inds_l1]
            x_l2_masked = x_l2[mask_inds_l2]
            x_l3_masked = x_l3[mask_inds_l3]
            x_l4_masked = x_l4[mask_inds_l4]
            # (L, K)

            y_l1_masked = y_l1[mask_inds_l1]
            y_l2_masked = y_l2[mask_inds_l2]
            y_l3_masked = y_l3[mask_inds_l3]
            y_l4_masked = y_l4[mask_inds_l4]
            # (L, K)

            return (x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked,
                    x_l1, mask_l1, y_l1, x_l1_masked, y_l1_masked,
                    x_l2, mask_l2, y_l2, x_l2_masked, y_l2_masked,
                    x_l3, mask_l3, y_l3, x_l3_masked, y_l3_masked,
                    x_l4, mask_l4, y_l4, x_l4_masked, y_l4_masked,)

    def resample_swnet(self, x_swnet_masked, y_swnet_masked):
        """

        :param x_swnet_masked: (L, 3)
        :param y_swnet_masked: (L, )
        :return:
        """

        x_swnet_g = x_swnet_masked[y_swnet_masked == 0]
        x_swnet_w = x_swnet_masked[(y_swnet_masked == 1) | (y_swnet_masked == 2)]

        y_swnet_g = y_swnet_masked[y_swnet_masked == 0]
        y_swnet_w = y_swnet_masked[(y_swnet_masked == 1) | (y_swnet_masked == 2)]

        perm_g = torch.randperm(x_swnet_g.size(0))
        perm_w = torch.randperm(x_swnet_w.size(0))

        min_size = min(perm_g.size(0), perm_w.size(0))
        idx_g = perm_g[:min_size]
        idx_w = perm_w[:min_size]

        x_swnet_g = x_swnet_g[idx_g]
        x_swnet_w = x_swnet_w[idx_w]
        y_swnet_g = y_swnet_g[idx_g]
        y_swnet_w = y_swnet_w[idx_w]

        new_x_swnet = torch.cat([x_swnet_g, x_swnet_w], dim=0)
        new_y_swnet = torch.cat([y_swnet_g, y_swnet_w], dim=0)

        return new_x_swnet, new_y_swnet

    def training_step(self, batch, batch_idx):
        result = self.process4tvt(batch, batch_idx)
        x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked = result[:7]
        if self.hparams.use_kriging_loss:
            loss_kriging = result[-1]

        loss_fdr = F.cross_entropy(x_fdr_masked, y_fdr_masked)
        acc_fdr, kc_fdr, iou_fdr, f1_fdr = self.metrics(x_fdr_masked.max(dim=1)[1], y_fdr_masked,
                                                        num_classes=9, remove_bg=False)
        # loss_swnet = F.cross_entropy(x_swnet_masked, y_swnet_masked)
        new_x_swnet, new_y_swnet = self.resample_swnet(x_swnet_masked, y_swnet_masked)
        if reduce(lambda x, y: x * y, new_x_swnet.size()) == 0 or reduce(lambda x, y: x * y, new_y_swnet.size()) == 0:
            no_swnet = True
        else:
            no_swnet = False
            loss_swnet = F.cross_entropy(new_x_swnet, new_y_swnet)
            acc_swnet, kc_swnet, iou_swnet, f1_swnet = self.metrics(new_x_swnet.max(dim=1)[1], new_y_swnet,
                                                                    num_classes=3, remove_bg=True)


        # acc_swnet, kc_swnet, iou_swnet = self.metrics(x_swnet_masked.max(dim=1)[1], y_swnet_masked,
        #                                               num_classes=3, remove_bg=True)
        # acc_swnet, kc_swnet, iou_swnet = self.metrics(new_x_swnet.max(dim=1)[1], new_y_swnet,
        #                                               num_classes=3, remove_bg=True)

        if self.hparams.use_model != 'lhn_unet':
            logs = {
                # 'train/loss': loss_fdr + loss_swnet,
                'train/loss_fdr': loss_fdr,
                # 'train/loss_swnet': loss_swnet,
                'train/acc_fdr': acc_fdr,
                'train/kc_fdr': kc_fdr,
                'train/iou_fdr': iou_fdr,
                # 'train/acc_swnet': acc_swnet,
                # 'train/kc_swnet': kc_swnet,
                # 'train/iou_swnet': iou_swnet,
            }

            if no_swnet:
                logs['train/loss'] = loss_fdr
            else:
                logs['train/loss'] = loss_fdr + loss_swnet
                logs['train/loss_swnet'] = loss_swnet
                logs['train/acc_swnet'] = acc_swnet
                logs['train/kc_swnet'] = kc_swnet
                logs['train/iou_swnet'] = iou_swnet

            if not self.hparams.use_kriging_loss:
                if no_swnet:
                    return {'loss': loss_fdr, 'log': logs}
                else:
                    return {'loss': loss_fdr + loss_swnet, 'log': logs}
            else:
                pass
                # logs['train/loss_kriging'] = loss_kriging
                # return {'loss': loss_fdr + loss_swnet + loss_kriging, 'log': logs}
        else:
            pass
            # need refine
            # (x_l1, mask_l1, y_l1, x_l1_masked, y_l1_masked,
            #  x_l2, mask_l2, y_l2, x_l2_masked, y_l2_masked,
            #  x_l3, mask_l3, y_l3, x_l3_masked, y_l3_masked,
            #  x_l4, mask_l4, y_l4, x_l4_masked, y_l4_masked) = result[7:]
            #
            # l1_ratios = [0.17952380655508302, 0.1526176884697518, 0.14403885487248097, 0.11961732655522918,
            #              0.11137710787313852, 0.10414634372605272, 0.10002826728467606, 0.08865060466358786]
            # l2_ratios = [0.19787245606635687, 0.1653361489975164, 0.1481186939861665, 0.12494729509967492,
            #              0.10725166425789422, 0.09160497777845729, 0.08706996670690959, 0.07779879710702424]
            # l3_ratios = [0.21941972939135929, 0.1751697829144009, 0.16413274465241395, 0.1241514198290188,
            #              0.10002689811695671, 0.07823622848404577, 0.07297882035155329, 0.06588437626025133]
            # l4_ratios = [0.25676770211339056, 0.18359423068983222, 0.1748901159437752, 0.11838090148346687,
            #              0.09029915136200042, 0.06357481030380548, 0.05813407443378186, 0.05435901366994727]
            # # based on the explained variance ratio of the PCAs
            #
            # loss_l1 = 0
            # for i, ratio in enumerate(l1_ratios):
            #     loss_l1 = loss_l1 + ratio * F.mse_loss(x_l1_masked[:, i], y_l1_masked[:, i])
            # loss_l2 = 0
            # for i, ratio in enumerate(l2_ratios):
            #     loss_l2 = loss_l2 + ratio * F.mse_loss(x_l2_masked[:, i], y_l2_masked[:, i])
            # loss_l3 = 0
            # for i, ratio in enumerate(l3_ratios):
            #     loss_l3 = loss_l3 + ratio * F.mse_loss(x_l3_masked[:, i], y_l3_masked[:, i])
            # loss_l4 = 0
            # for i, ratio in enumerate(l4_ratios):
            #     loss_l4 = loss_l4 + ratio * F.mse_loss(x_l4_masked[:, i], y_l4_masked[:, i])
            #
            # loss_level = (loss_l1 + loss_l2 + loss_l3 + loss_l4) * 10
            #
            # logs = {
            #     'train/loss': loss_fdr + loss_swnet + loss_level,
            #     'train/loss_level': loss_level,
            #     'train/loss_fdr': loss_fdr,
            #     'train/loss_swnet': loss_swnet,
            #     'train/acc_fdr': acc_fdr,
            #     'train/kc_fdr': kc_fdr,
            #     'train/iou_fdr': iou_fdr,
            #     'train/acc_swnet': acc_swnet,
            #     'train/kc_swnet': kc_swnet,
            #     'train/iou_swnet': iou_swnet,
            # }
            #
            # return {'loss': loss_fdr + loss_swnet + loss_level,
            #         'log': logs}


    def validation_step(self, batch, batch_idx):
        x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked = self.process4tvt(batch, batch_idx)[:7]

        loss_fdr = F.cross_entropy(x_fdr_masked, y_fdr_masked)
        acc_fdr, kc_fdr, iou_fdr, f1_fdr, precision_fdr, recall_fdr = self.metrics(x_fdr_masked.max(dim=1)[1], y_fdr_masked,
                                                        num_classes=9, remove_bg=False)
        # loss_swnet = F.cross_entropy(x_swnet_masked, y_swnet_masked)
        new_x_swnet, new_y_swnet = self.resample_swnet(x_swnet_masked, y_swnet_masked)
        if reduce(lambda x, y: x * y, new_x_swnet.size()) == 0 or reduce(lambda x, y: x * y, new_y_swnet.size()) == 0:
            no_swnet = True
        else:
            no_swnet = False
            loss_swnet = F.cross_entropy(new_x_swnet, new_y_swnet)
            # acc_swnet, kc_swnet, iou_swnet = self.metrics(x_swnet_masked.max(dim=1)[1], y_swnet_masked,
            #                                               num_classes=3, remove_bg=True)
            acc_swnet, kc_swnet, iou_swnet, f1_swnet, precision_swnet, recall_swnet = self.metrics(new_x_swnet.max(dim=1)[1], new_y_swnet,
                                                                    num_classes=3, remove_bg=True, is_swnet=True)


        if no_swnet:
            return {'val_loss': loss_fdr,
                    'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr, 'f1_fdr': f1_fdr, 'precision_fdr': precision_fdr,
                    'recall_fdr': recall_fdr,}
        else:
            return {'val_loss': loss_fdr + loss_swnet,
                    'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr, 'f1_fdr': f1_fdr, 'precision_fdr': precision_fdr,
                    'recall_fdr': recall_fdr,
                    'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet, 'iou_swnet': iou_swnet, 'f1_swnet': f1_swnet,
                    'precision_swnet': precision_swnet, 'recall_swnet': recall_swnet}

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
        x, mask, y, x_fdr_masked, x_swnet_masked, y_fdr_masked, y_swnet_masked = self.process4tvt(batch, batch_idx)[:7]

        # loss_fdr = F.nll_loss(torch.log(x_fdr_masked), y_fdr_masked)
        # loss_swnet = F.nll_loss(torch.log(x_swnet_masked), y_swnet_masked)
        loss_fdr = F.cross_entropy(x_fdr_masked, y_fdr_masked)
        # loss_swnet = F.cross_entropy(x_swnet_masked, y_swnet_masked)
        acc_fdr, kc_fdr, iou_fdr, f1_fdr, precision_fdr, recall_fdr = self.metrics(x_fdr_masked.max(dim=1)[1], y_fdr_masked,
                                                       num_classes=9, remove_bg=False)

        new_x_swnet, new_y_swnet = self.resample_swnet(x_swnet_masked, y_swnet_masked)
        if reduce(lambda x, y: x * y, new_x_swnet.size()) == 0 or reduce(lambda x, y: x * y, new_y_swnet.size()) == 0:
            no_swnet = True
        else:
            no_swnet = False
            loss_swnet = F.cross_entropy(new_x_swnet, new_y_swnet)
            # acc_swnet, kc_swnet, iou_swnet = self.metrics(x_swnet_masked.max(dim=1)[1], y_swnet_masked,
            #                                               num_classes=3, remove_bg=True)
            acc_swnet, kc_swnet, iou_swnet, f1_swnet, precision_swnet, recall_swnet = self.metrics(new_x_swnet.max(dim=1)[1], new_y_swnet,
                                                                    num_classes=3, remove_bg=True, is_swnet=True)
        # loss_swnet = F.cross_entropy(new_x_swnet, new_y_swnet)

        # acc_swnet, kc_swnet, iou_swnet = self.metric(x_swnet_masked.max(dim=1)[1], y_swnet_masked,
        #                                              num_classes=3, remove_bg=True)
        # acc_swnet, kc_swnet, iou_swnet = self.metrics(new_x_swnet.max(dim=1)[1], new_y_swnet,
        #                                               num_classes=3, remove_bg=True)

        # return {'val_loss': loss_fdr + loss_swnet,
        #         'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr,
        #         'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet, 'iou_swnet': iou_swnet}
        if no_swnet:
            return {'test_loss': loss_fdr,
                    'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr, 'f1_fdr': f1_fdr, 'precision_fdr': precision_fdr,
                    'recall_fdr': recall_fdr}
        else:
            return {'test_loss': loss_fdr + loss_swnet,
                    'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr, 'f1_fdr': f1_fdr, 'precision_fdr': precision_fdr,
                    'recall_fdr': recall_fdr,
                    'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet, 'iou_swnet': iou_swnet, 'f1_swnet': f1_swnet,
                    'precision_swnet': precision_swnet, 'recall_swnet': recall_swnet
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
                new_logs['test/'+key] = torch.stack(logs[key]).mean()
            else:
                new_logs = {**new_logs, **self.unpack_multi_class('test/' + key, torch.stack(logs[key]).mean(0))}

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
            'x', 'slope', 'curvature', 'd8'
        ]

        for arg in arg_to_device_list:
            try:
                batch[arg] = batch[arg].to(self.device)
            except KeyError:
                pass

        x = batch['x'].to(self.device)
        if self.hparams.use_slope:
            x = torch.cat([x, batch['slope']], dim=1)

        if self.hparams.use_curvature:
            x = torch.cat([x, batch['curvature']], dim=1)

        if self.hparams.use_slope and self.hparams.use_curvature:
            x = torch.cat([x, batch['slope'] ** 2 * batch['curvature']], dim=1)

        if self.hparams.use_d8:
            d8 = self.d8_emb(batch['d8'].squeeze(1)).permute(0, 3, 1, 2)
            x = torch.cat([x, d8], dim=1)

        mask = batch['mask']
        y = batch['y']

        result = self(x)

        x_fdr = result['x_fdr'].permute(0, 2, 3, 1).argmax(3)
        # (B, H, W, 9)
        x_swnet_logits = result['x_swnet'].permute(0, 2, 3, 1)
        x_swnet = x_swnet_logits.argmax(3)
        # (B, H, W, 3)

        y_fdr = y[:, 0]
        y_swnet = y[:, 1]
        # (B, H, W)

        # get score
        mask_inds = mask == 255
        x_fdr_masked = x_fdr[mask_inds]
        # (L,)
        x_swnet_masked = x_swnet[mask_inds]
        # (L,)
        y_fdr_masked = y_fdr[mask_inds]
        # (L, )
        y_swnet_masked = y_swnet[mask_inds]
        # (L, )
        acc_fdr, kc_fdr, iou_fdr, f1_fdr, precision_fdr, recall_fdr = self.metrics(x_fdr_masked.cpu(), y_fdr_masked.cpu(),
                                                        num_classes=9, remove_bg=False)
        new_x_swnet, new_y_swnet = self.resample_swnet(x_swnet_masked, y_swnet_masked)
        if reduce(lambda x, y: x * y, new_x_swnet.size()) == 0 or reduce(lambda x, y: x * y, new_y_swnet.size()) == 0:
            no_swnet = True
        else:
            no_swnet = False
            acc_swnet, kc_swnet, iou_swnet, f1_swnet, precision_swnet, recall_swnet = self.metrics(new_x_swnet.cpu(), new_y_swnet.cpu(),
                                                                    num_classes=3, remove_bg=True, is_swnet=True)
        metrics = {
            'acc_fdr': acc_fdr, 'kc_fdr': kc_fdr, 'iou_fdr': iou_fdr, 'f1_fdr': f1_fdr, 'precision_fdr': precision_fdr,
                    'recall_fdr': recall_fdr,
        }
        if not no_swnet:
            metrics = {
                **metrics, **{
                    'acc_swnet': acc_swnet, 'kc_swnet': kc_swnet, 'iou_swnet': iou_swnet, 'f1_swnet': f1_swnet,
                    'precision_swnet': precision_swnet, 'recall_swnet': recall_swnet
                }
            }

        new_metrics = {}
        for key in metrics:
            if 'precision' not in key and 'recall' not in key and 'f1' not in key:
                new_metrics[key] = metrics[key]
            else:
                new_metrics = {**new_metrics, **self.unpack_multi_class(key, metrics[key])}


        return x_fdr.squeeze(0), x_swnet.squeeze(0), x_swnet_logits.squeeze(0),\
               y_fdr.squeeze(0), y_swnet.squeeze(0), mask.squeeze(0),\
               batch['d8'].squeeze(0), new_metrics


    def infer_simple(self, batch):
        arg_to_device_list = [
            'x', 'd8'
        ]

        for arg in arg_to_device_list:
            try:
                batch[arg] = batch[arg].to(self.device)
            except KeyError:
                pass

        x = batch['x'].to(self.device)

        if self.hparams.use_d8:
            d8 = self.d8_emb(batch['d8'].squeeze(1)).permute(0, 3, 1, 2)
            x = torch.cat([x, d8], dim=1)

        with torch.no_grad():
            result = self(x)

        x_fdr = result['x_fdr'].permute(0, 2, 3, 1).argmax(3)
        # (B, H, W, 9)
        x_swnet_logits = result['x_swnet'].permute(0, 2, 3, 1)
        x_swnet_logits = torch.softmax(x_swnet_logits, dim=3)
        x_swnet = x_swnet_logits.argmax(3)
        # (B, H, W, 3)

        return x_fdr.squeeze(0), x_swnet.squeeze(0), x_swnet_logits.squeeze(0),\
               batch['d8'].squeeze(0),


    def metrics(self, pred, target, num_classes=3, remove_bg=False, is_swnet=False):
        if is_swnet:
            pred[pred == 2] = 1
            target[target == 2] = 1
            num_classes = 2

        confusion_m = mf.confusion_matrix(pred, target, num_classes=num_classes)

        # acc
        accuracy = confusion_m.diag().sum() / len(pred)

        # kappa
        p0 = accuracy
        pc = 0
        for i in range(confusion_m.shape[0]):
            pc = pc + confusion_m[i].sum() * confusion_m[:, i].sum()
        pc = pc / len(pred)**2
        kc = (p0 - pc) / (1 - pc)

        # iou
        if remove_bg:
            iou = mf.iou(pred, target, num_classes=num_classes, ignore_index=0)
        else:
            iou = mf.iou(pred, target, num_classes=num_classes)

        f1 = mf.f1_score(pred, target, num_classes=num_classes, class_reduction='none')
        precision = mf.precision(pred, target, num_classes=num_classes, class_reduction='none')
        recall = mf.recall(pred, target, num_classes=num_classes, class_reduction='none')

        return accuracy, kc, iou, f1, precision, recall
