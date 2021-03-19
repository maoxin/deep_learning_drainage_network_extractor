from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
import torch


class SPSegNet(nn.Module):
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
        parser.add_argument('--drop-rate', type=float, default=0.5)
        parser.fromfile_prefix_chars = "@"

        return parser

    def __init__(self, num_classes, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SPSegNet, self).__init__()

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
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Sequential(nn.Conv2d(filter_config[0] * 4, filter_config[0], 3, 1, 1),
                                        nn.Conv2d(filter_config[0], num_classes, 3, 1, 1))

    def forward_(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return feat

    def forward(self, x):
        x_sub2 = F.avg_pool2d(x, 2)
        x_sub4 = F.avg_pool2d(x, 4)
        x_sub8 = F.avg_pool2d(x, 8)

        x = self.forward_(x)
        x_sub2 = self.forward_(x_sub2)
        x_sub4 = self.forward_(x_sub4)
        x_sub8 = self.forward_(x_sub8)

        x = torch.cat([
            x,
            F.interpolate(x_sub2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(x_sub4, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(x_sub8, scale_factor=8, mode='bilinear', align_corners=True),
        ], dim=1)

        return self.classifier(x)


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
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
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()
        self.n_blocks = n_blocks

        self.l1 = nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1)
        self.l1_ = nn.Sequential(
            nn.BatchNorm2d(n_in_feat),
            nn.ReLU(inplace=True)
        )

        if n_blocks > 1:
            layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                      nn.BatchNorm2d(n_out_feat),
                      nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]


        if n_blocks > 1:
            self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        x1 = self.l1(unpooled)
        x1 = self.l1_(x1)
        if self.n_blocks > 1:
            x1 = self.features(x1)

        return x1