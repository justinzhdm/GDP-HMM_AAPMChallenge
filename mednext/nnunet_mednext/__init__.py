from __future__ import absolute_import
# print("\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
#       "\"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.\" "
#       "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n")
# print("If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n")

from . import *
# from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
# from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
# from nnunet_mednext.network_architecture.mednextv1.blocks import \
#     MedNeXtBlock, MedNeXtUpBlock, MedNeXtDownBlock
# from nnunet_mednext.run.load_weights import upkern_load_weights

from mednext.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from mednext.nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
from mednext.nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtUpBlock, MedNeXtDownBlock
from mednext.nnunet_mednext.run.load_weights import upkern_load_weights

