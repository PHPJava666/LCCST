from networks.unet import UNet, UNet_CML
from networks.VNet import VNet, VNet_CML

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "unet_CML":
        net = UNet_CML(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if net_type == "VNet_CML" and mode == "train":
        net = VNet_CML(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet_CML" and mode == "test":
        net = VNet_CML(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    # if net_type == "unet_CML" and mode == "train":
    #     net = UNet_CML(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    # if net_type == "unet_CML" and mode == "test":
    #     net = UNet_CML(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
