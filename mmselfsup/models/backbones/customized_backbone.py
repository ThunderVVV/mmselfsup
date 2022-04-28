import torch.nn as nn
from einops import rearrange
from easydict import EasyDict

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor,
)
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention

from ..builder import BACKBONES

@BACKBONES.register_module()
class CustomizedBackbone(nn.Module):
    def __init__(self, SelfSL_layer=False, instance_map="all"):
        super(CustomizedBackbone, self).__init__()
        
        opt=EasyDict()
        opt.Transformation="None"
        opt.FeatureExtraction="ResNet"
        opt.SequenceModeling="BiLSTM"
        opt.Prediction="None"
        opt.num_fiducial=20
        opt.imgH=32
        opt.imgW=100
        opt.input_channel=3
        opt.output_channel=512
        opt.hidden_size=256
        opt.self="MoCoSeqCLR"
        opt.instance_map=instance_map
        opt.moco_dim=128
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }
        self.SelfSL_layer = SelfSL_layer

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        if not SelfSL_layer or SelfSL_layer == "CNNLSTM":  # for STR or CNNLSTM SSL
            """Sequence modeling"""
            if opt.SequenceModeling == "BiLSTM":
                self.SequenceModeling = nn.Sequential(
                    BidirectionalLSTM(
                        self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                    ),
                    BidirectionalLSTM(
                        opt.hidden_size, opt.hidden_size, opt.hidden_size
                    ),
                )
                self.SequenceModeling_output = opt.hidden_size
            else:
                print("No SequenceModeling module specified")
                self.SequenceModeling_output = self.FeatureExtraction_output

        if not SelfSL_layer:  # for STR.
            """Prediction"""
            raise Exception("Prediction is not needed")

        else:
            """for self-supervised learning (SelfSL)"""
            if self.opt.self == "RotNet" or self.opt.self == "MoCo":
                self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1
            elif self.opt.self == "MoCoSeqCLR":
                if self.opt.instance_map == "window":
                    self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 4))  # make width -> 5 instances
                elif self.opt.instance_map == "all":
                    self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))  # make width -> 1 instances
            else:
                raise NotImplementedError
            if SelfSL_layer == "CNN":
                self.SelfSL_FFN_input = self.FeatureExtraction_output
            if SelfSL_layer == "CNNLSTM":
                self.SelfSL_FFN_input = self.SequenceModeling_output

            # if "MoCo" in self.opt.self:
            #     self.fc = nn.Linear(
            #         self.SelfSL_FFN_input, opt.moco_dim
            #     )  # 128 is used for MoCo paper.
            

    def forward(self, image):
        SelfSL_layer = self.SelfSL_layer
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ for self supervised learning on Feature extractor (CNN part) """
        if SelfSL_layer == "CNN":
            visual_feature = visual_feature.permute(0, 2, 1)  # [b, w, c] -> [b, c, w]
            visual_feature = self.AdaptiveAvgPool_2(
                visual_feature
            )  # [b, c, w] -> [b, c, t]
            visual_feature = rearrange(visual_feature, 'b c t -> (b t) c')
            visual_feature = visual_feature[:, :, None, None] # -> bchw
            # visual_feature = visual_feature.squeeze(2)  # [b, c, 1] -> [b, c]
            # prediction_SelfSL = self.fc(
            #     visual_feature
            # )  # [b, c] -> [b, SelfSL_class]
            return tuple([visual_feature,])

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(
                visual_feature
            )  # [b, num_steps, opt.hidden_size]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if SelfSL_layer == "CNNLSTM":
            contextual_feature = contextual_feature.permute(0, 2, 1)  # [b, w, c] -> [b, c, w]
            contextual_feature = self.AdaptiveAvgPool_2(
                contextual_feature
            )  # [b, c, w] -> [b, c, t]
            contextual_feature = rearrange(contextual_feature, 'b c t -> (b t) c')
            contextual_feature = contextual_feature[:, :, None, None]   # -> bchw
            # contextual_feature = contextual_feature.squeeze(2)  # [b, c, 1] -> [b, c]
            # prediction_SelfSL = self.fc(
            #     contextual_feature
            # )  # [b, c] -> [b, SelfSL_class]
            return tuple([contextual_feature,])
