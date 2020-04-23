from __future__ import absolute_import
import torch

from .osnet import *
from .osnet_ain import *

from .osnet_mod import *
from .osnet_mod_two_branch import *
from .osnet_mod_two_multi import *
from .resnet_mod_two_branch import *
from .osnet_mod_two_branch_a import *
from .osnet_mod_single import *
from .osnet_mod_parts import *
from .osnet_bnneck import *
from .osnet_bnneck_multi import *
from .resnet_ori import *
from .pcb import *
from .mlfn import *
__model_factory = {
    # image classification models
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_x1_0_mod': osnet_x1_0_mod,
    'osnet_mod_two_branch': osnet_mod_two_branch,
    'osnet_mod_two_branch_a': osnet_mod_two_branch_a,
    'osnet_mod_two_multi': osnet_mod_two_multi,
    'resnet_mod_two_branch': resnet_mod_two_branch,
    'osnet_mod_single': osnet_mod_single,
    'osnet_mod_parts': osnet_mod_parts,
    'osnet_bnneck': osnet_bnneck,
    'osnet_bnneck_multi': osnet_bnneck_multi,
    'resnet_ori': resnet_ori,
    'pcb_p4': pcb_p4,
    'pcb_p6': pcb_p6,
    'mlfn': mlfn,
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name,
                num_classes,
                loss='softmax',
                pretrained=True,
                with_attention=True,
                use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(
            name, avai_models))
    return __model_factory[name](num_classes=num_classes,
                                 loss=loss,
                                 pretrained=pretrained,
                                 with_attention=with_attention,
                                 use_gpu=use_gpu)
