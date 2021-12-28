import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        #self.opt = opt
        self.stages = {'Trans': 'TPS', 'Feat': 'ResNet',
                       'Seq': 'BiLSTM', 'Pred': 'Attn'}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32,100), I_r_size=(32,100), I_channel_num=1)


        """ FeatureExtraction====Resnet """
        self.FeatureExtraction = ResNet_FeatureExtractor(1,512)
        
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256


        """ Prediction==Attn """
        self.Prediction = Attention(self.SequenceModeling_output, 256, 38) #512,256,
        

    def forward(self, input, text, is_train=True):
        # print("input....",input.shape)
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        # print("Trans....",input.shape)
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # print("FeatureExtraction....",visual_feature.shape)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        # print("visual_feature....",visual_feature.shape)
        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)
        # print("contextual_feature....",contextual_feature.shape)
        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)
        # print("prediction....",prediction.shape)

        return prediction
