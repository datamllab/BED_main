import torch.nn as nn
import torch

# from ai8x_yolov1 import FusedConv2dBNReLU, FusedConv2dBNLeakyReLU
# from ai8x_yolov1 import FusedMaxPoolConv2dBNReLU, FusedMaxPoolConv2dBNLeakyReLU
# from ai8x_yolov1 import FusedLinearSigmoid

import sys
sys.path.append("../../")

from ai8x import FusedLinearReLU, Linear, FusedConv1dReLU, FusedConv1dBNReLU, Conv1d, Conv2d
from ai8x import FusedConv2dReLU, FusedMaxPoolConv2dReLU, FusedConv2dBNReLU, FusedMaxPoolConv2dBNReLU
from ai8x import QuantizationAwareModule

class Yolov1_net(nn.Module):

    def __init__(self, B=2, num_classes=5, bias=False, **kwargs):
        super().__init__()
        print("YOLO V1 Model_Z %d class (224 input), %d bounding boxes.".format(num_classes, B))
        self.B = B
        self.Classes_Num = num_classes

        # self.conv_448_224 = FusedMaxPoolConv2dReLU(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, **kwargs)

        self.Conv_224 = FusedConv2dReLU(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        # Convention(3, 64, 3, 1, 1),
        # nn.MaxPool2d(2, 2),
        self.Conv_112 = FusedMaxPoolConv2dReLU(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        # Convention(64, 24, 3, 1, 1),
        # nn.MaxPool2d(2, 2),
        self.Conv_56_1 = FusedMaxPoolConv2dReLU(in_channels=24, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.Conv_56_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_56_3 = FusedConv2dBNReLU(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_56_4 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        # Convention(24, 16, 1, 1, 0),
        # Convention(16, 32, 3, 1, 1),
        # Convention(32, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # nn.MaxPool2d(2, 2),
        self.Conv_28_1 = FusedMaxPoolConv2dBNReLU(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_28_2 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_28_3 = FusedConv2dBNReLU(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_28_4 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_28_5 = FusedConv2dBNReLU(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_28_6 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        # Convention(64, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # Convention(64, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # Convention(64, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # nn.MaxPool2d(2, 2),
        self.Conv_14_1 = FusedMaxPoolConv2dBNReLU(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_14_2 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_14_3 = FusedConv2dBNReLU(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_14_4 = FusedConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_14_5 = FusedConv2dBNReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_14_6 = FusedConv2dBNReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        # Convention(64, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # Convention(64, 32, 1, 1, 0),
        # Convention(32, 64, 3, 1, 1),
        # Convention(64, 64, 3, 1, 1),
        # Convention(64, 64, 3, 2, 1),
        self.Conv_7_1 = FusedMaxPoolConv2dBNReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_7_2 = FusedConv2dBNReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        # Convention(64, 64, 3, 1, 1),
        # Convention(64, 64, 3, 1, 1),
        self.Conv_Res_1 = FusedConv2dBNReLU(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_2 = FusedConv2dBNReLU(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_3 = FusedConv2dBNReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_4 = Conv2d(in_channels=16, out_channels=self.B * 5 + self.Classes_Num, kernel_size=1, stride=1, padding=0, bias=True, wide=True, **kwargs)
        # Convention(64, 64, 1, 1, 0),  # 1 * 1
        # Convention(64, 16, 1, 1, 0),
        # Convention(16, 16, 1, 1, 0),  # 1 * 1
        # nn.Conv2d(16, self.B * 5 + self.Classes_Num, 1, 1, 0),
        # nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x): # forward not influence the board

        x = self.Conv_224(x)
        #print("First Layer:",x)
        x = self.Conv_112(x)
        # import numpy as np
        # print("Second Layer: Unique Values:->", np.unique(x.detach().reshape(-1).numpy().astype(np.int)),"Complete Output: ",x)
        x = self.Conv_56_1(x)
        #print("First Layer:",x)
        x = self.Conv_56_2(x)
        #print("First Layer:",x)
        x = self.Conv_56_3(x)
        x = self.Conv_56_4(x)
        x = self.Conv_28_1(x)
        x = self.Conv_28_2(x)
        x = self.Conv_28_3(x)
        x = self.Conv_28_4(x)
        x = self.Conv_28_5(x)
        x = self.Conv_28_6(x)
        x = self.Conv_14_1(x)
        x = self.Conv_14_2(x)
        x = self.Conv_14_3(x)
        x = self.Conv_14_4(x)
        x = self.Conv_14_5(x)
        x = self.Conv_14_6(x)
        x = self.Conv_7_1(x)
        x = self.Conv_7_2(x)
        x = self.Conv_Res_1(x)
        x = self.Conv_Res_2(x)
        x = self.Conv_Res_3(x)
        #print("Final Conv:", x)
        x_fl_output = self.Conv_Res_4(x)
        x = x_fl_output.permute(0, 2, 3, 1)
        class_possible = torch.softmax(x[:, :, :, 10:], dim=3)
        x = torch.cat((torch.sigmoid(x[:,:,:,0:10]), class_possible), dim=3)

        return x, x_fl_output

    def quantize_layer(self, layer_index=None, qat_policy=None):

        if layer_index is None or qat_policy is None:
            return

        # print(layer_index)
        layer_buf = list(self.children())
        layer = layer_buf[layer_index]
        layer.init_module(qat_policy['weight_bits'], qat_policy['bias_bits'], True)

    def fuse_bn_layer(self, layer_index=None):

        if layer_index is None:
            return

        # print(layer_index)
        layer_buf = list(self.children())
        layer = layer_buf[layer_index]
        if isinstance(layer, QuantizationAwareModule) and layer.bn is not None:
            w = layer.op.weight.data
            b = layer.op.bias.data
            device = w.device

            r_mean = layer.bn.running_mean
            r_var = layer.bn.running_var
            r_std = torch.sqrt(r_var + 1e-20)
            beta = layer.bn.weight
            gamma = layer.bn.bias

            if beta is None:
                beta = torch.ones(w.shape[0]).to(device)
            if gamma is None:
                gamma = torch.zeros(w.shape[0]).to(device)

            w_new = w * (beta / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
            b_new = (b - r_mean) / r_std * beta + gamma

            layer.op.weight.data = w_new
            layer.op.bias.data = b_new
            layer.bn = None


def yolov1_net(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return Yolov1_net(**kwargs)


models = [
    {
        'name': 'ai85net5',
        'min_input': 1,
        'dim': 2,
    },
]
