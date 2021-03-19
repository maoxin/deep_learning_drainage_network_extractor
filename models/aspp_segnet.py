from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.kriging.kriging_loss import KrigingLoss


class ASPPSegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.fromfile_prefix_chars = "@"

        return parser

    def __init__(self, num_classes, n_init_features=1,
                 filter_config=(64, 128, 256, 512, 512), use_kriging_loss=False):
        super(ASPPSegNet, self).__init__()
        self.use_kriging_loss = use_kriging_loss

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 2, 2, 2)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (2, 2, 2, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i],
                                          self.use_kriging_loss))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        loss_kriging = 0
        for i in range(0, 5):
            result = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])
            if self.use_kriging_loss:
                feat, kl = result
                loss_kriging = loss_kriging + kl
            else:
                feat = result[0]
        loss_kriging = loss_kriging / 5

        if not self.use_kriging_loss:
            return self.classifier(feat)
        else:
            return self.classifier(feat), loss_kriging


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False,
                                     groups=planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, use_kriging_loss=False):
        super(_Decoder, self).__init__()
        self.use_kriging_loss = use_kriging_loss
        self.n_blocks = n_blocks

        self.aspp1 = _ASPPModule(n_in_feat, n_in_feat, 3, padding=1, dilation=1)
        self.aspp2 = _ASPPModule(n_in_feat, n_in_feat, 3, padding=12, dilation=12)
        self.aspp3 = _ASPPModule(n_in_feat, n_in_feat, 3, padding=24, dilation=24)
        self.aspp4 = _ASPPModule(n_in_feat, n_in_feat, 3, padding=36, dilation=36)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(n_in_feat, n_in_feat, 1, stride=1, bias=False,
                                                       groups=n_in_feat),
                                             nn.BatchNorm2d(n_in_feat),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(n_in_feat, n_in_feat, 1, bias=False, groups=n_in_feat)
        self.bn1 = nn.BatchNorm2d(n_in_feat)
        self.relu1 = nn.ReLU(inplace=True)

        if self.use_kriging_loss:
            self.kl = KrigingLoss(sample_size=64)

        if n_blocks > 1:
            self.conv2 = nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(n_out_feat)
            self.relu2 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)

        x_ = self.aspp1(unpooled)
        x_ = self.aspp2(unpooled) + x_
        x_ = self.aspp3(unpooled) + x_
        x_ = self.aspp4(unpooled) + x_
        x_ = F.interpolate(self.global_avg_pool(unpooled), size=x_.size()[2:], mode='bilinear',
                           align_corners=True) + x_

        x_ = self.conv1(x_)
        x_ = self.bn1(x_)
        x_ = self.relu1(x_)

        if self.use_kriging_loss:
            loss_kriging = self.kl(x, indices, x_)

        if self.n_blocks > 1:
            x_ = self.relu2(self.bn2(self.conv2(x_)))

        if self.use_kriging_loss:
            x_ = (x_, loss_kriging)
        else:
            x_ = (x_, )

        return x_

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()