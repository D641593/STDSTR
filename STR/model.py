import torch.nn as nn
import torch
from reAttn import *
from positionAttn import *

class clsScore(nn.Module):
    def __init__(self, num_class,max_length):
        super(clsScore,self).__init__()
        self.backbone = ResNetAttn()
        self.max_length = max_length
        self.out_channels = 512
        self.attn = positionAttention(max_length=max_length)
        self.cls = nn.Linear(self.out_channels, num_class)

    def forward(self, x):
        x = self.backbone(x)
        attn_vec, attn = self.attn(x)
        cls_score = self.cls(attn_vec)
        cls_score = nn.functional.log_softmax(cls_score,dim = -1)
        return cls_score

if __name__ == '__main__':
    model = clsScore(1000)
    sample = torch.zeros(8,3,32,512)
    output = model(sample)
    print(output.shape)
