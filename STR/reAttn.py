import torch.nn as nn
import resnet
from positionEncoding import *

img_except_h, img_except_w = 32,512
class ResNetAttn(nn.Module):
    def __init__(self):
        super(ResNetAttn,self).__init__()
        self.resnet = resnet.resnet50()
        self.d_model = 512
        nhead = 8
        dim_f = 2048
        dropout = 0.1
        activation = 'relu'
        num_layers = 4
        # print("ResNetAttn")
        self.position = PositionalEncoding(self.d_model,max_len=img_except_h//4 * img_except_w//4 )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = nhead,
            dim_feedforward = dim_f,
            dropout = dropout,
            activation = activation,)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers)

    def forward(self,x):
        # print(x.shape)
        x = self.resnet(x)
        # print(x.shape)
        n,c,h,w = x.shape
        x = x.view(n,c,-1).permute(2,0,1)
        x = self.position(x)
        x = self.encoder(x)
        x = x.permute(1,2,0).view(n,c,h,w)
        # print(x.shape)
        return x

if '__main__' == __name__:
    sample = torch.zeros(4,3,img_except_h, img_except_w)
    model = ResNetAttn()
    output = model(sample)
    # print(output.shape)

