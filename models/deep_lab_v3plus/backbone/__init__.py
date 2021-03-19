from models.deep_lab_v3plus.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, in_channels=3):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, in_channels=in_channels)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, in_channels=in_channels)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, in_channels=in_channels)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, in_channels=in_channels)
    else:
        raise NotImplementedError
