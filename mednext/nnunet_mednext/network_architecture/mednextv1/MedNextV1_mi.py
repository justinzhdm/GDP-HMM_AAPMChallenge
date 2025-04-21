import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# from nnunet_mednext.network_architecture.mednextv1.blocks import *
# from mednext.nnunet_mednext.network_architecture.mednextv1.blocks import *
from mednext.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


class MedNeXt_mi(nn.Module):

    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
    ):

        super().__init__()

        self.m1 = MedNeXt(
                            in_channels = in_channels,
                            n_channels = 32,
                            n_classes = n_classes,
                            exp_r=[3,4,8,8,8,8,8,4,3],
                            kernel_size=kernel_size,
                            deep_supervision=deep_supervision,
                            do_res=True,
                            do_res_up_down = True,
                            block_counts = [3,4,8,8,8,8,8,4,3],
                            checkpoint_style = 'outside_block'
                            )
        self.m2 = MedNeXt(
                            in_channels = in_channels,
                            n_channels = 32,
                            n_classes = n_classes,
                            exp_r=[3,4,8,8,8,8,8,4,3],
                            kernel_size=kernel_size,
                            deep_supervision=deep_supervision,
                            do_res=True,
                            do_res_up_down = True,
                            block_counts = [3,4,8,8,8,8,8,4,3],
                            checkpoint_style = 'outside_block'
                            )
        self.m3 = MedNeXt(
                            in_channels = in_channels,
                            n_channels = 32,
                            n_classes = n_classes,
                            exp_r=[3,4,8,8,8,8,8,4,3],
                            kernel_size=kernel_size,
                            deep_supervision=deep_supervision,
                            do_res=True,
                            do_res_up_down = True,
                            block_counts = [3,4,8,8,8,8,8,4,3],
                            checkpoint_style = 'outside_block'
                            )
        self.m4 = MedNeXt(
                            in_channels = in_channels,
                            n_channels = 32,
                            n_classes = n_classes,
                            exp_r=[3,4,8,8,8,8,8,4,3],
                            kernel_size=kernel_size,
                            deep_supervision=deep_supervision,
                            do_res=True,
                            do_res_up_down = True,
                            block_counts = [3,4,8,8,8,8,8,4,3],
                            checkpoint_style = 'outside_block'
                            )
        self.m_list = [self.m1, self.m2, self.m3, self.m4]

    def forward(self, x):
        outputs_list = []
        for model in self.m_list:
            outputs = model(x)
            outputs = torch.sigmoid(outputs)
            outputs = outputs * 12.
            outputs_list.append(outputs)
        outputs = torch.stack(outputs_list, 0)
        outputs = torch.sum(outputs, 0) / float(len(self.m_list))
        return outputs




