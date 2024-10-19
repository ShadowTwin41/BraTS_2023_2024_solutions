from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

import sys
sys.path.insert(1, '/homes/andre.ferreira/BraTS2023/nnFormer')
from nnformer.network_architecture.nnFormer_tumor import nnFormer
from torch import nn


class nnUNetTrainer_nnFormer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        #num_stages = len(configuration_manager.conv_kernel_sizes)

        #dim = len(configuration_manager.conv_kernel_sizes[0])
        #conv_op = convert_dim_to_conv_op(dim)

        #label_manager = plans_manager.get_label_manager(dataset_json)

        #segmentation_network_class_name = configuration_manager.UNet_class_name
        
        """
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        """
        #network_class = mapping[segmentation_network_class_name]

        #conv_or_blocks_per_stage = {
        #    'n_conv_per_stage'
        #    if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        #    'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        #}
        # network class name!!
        """
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        """
        embedding_dim=96
        depths=[2, 2, 2, 2]
        num_heads=[3, 6, 12, 24]
        embedding_patch_size=[4,4,4]
        window_size=[4,4,8,4]
        input_channels = 4
        conv_op = nn.Conv3d
        num_classes = 3 
        deep_supervision = True
        evaluation = True
        if evaluation:
            print("Set to evaluation mode. If not intended, change /homes/andre.ferreira/BraTS2023/nnUNet/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_nnFormer.py")
        else:
            print("Set to train mode. If not intended, change /homes/andre.ferreira/BraTS2023/nnUNet/nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainer_nnFormer.py")
        crop_size = [
                128,
                128,
                128
            ]
        print("#############_Using nnFormer_#############")
        model=nnFormer(crop_size=crop_size,
                                embedding_dim=embedding_dim,
                                input_channels=input_channels,
                                num_classes=num_classes,
                                conv_op=conv_op,
                                depths=depths,
                                num_heads=num_heads,
                                patch_size=embedding_patch_size,
                                window_size=window_size,
                                deep_supervision=deep_supervision,
                                evaluation=evaluation)
        return model
