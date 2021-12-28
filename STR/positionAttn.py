import torch.nn as nn
import torch
from positionEncoding import *

img_except_h, img_except_w = 32,512
def encoder(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True)
    )

def decoder(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1,scale_factor = None,size = None):
    return nn.Sequential(
        nn.Upsample(size = size, scale_factor=scale_factor,mode='nearest'),
        nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True)
    )

class positionAttention(nn.Module):
    def __init__(self,max_length,in_channel = 512,out_channel = 128,h = 8, w = 128):
        super(positionAttention,self).__init__()
        print("positionAttn")
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder(in_channel,out_channel,stride=(1,2)),
            encoder(out_channel,out_channel,stride=(2,2)),
            encoder(out_channel,out_channel,stride=(2,2)),
            encoder(out_channel,out_channel,stride=(2,2)),
        )
        self.k_decoder = nn.Sequential(
            decoder(out_channel,out_channel,scale_factor = 2),
            decoder(out_channel,out_channel,scale_factor = 2),
            decoder(out_channel,out_channel,scale_factor = 2),
            decoder(out_channel,in_channel,size = (h,w)),
        )
        self.position = PositionalEncoding(in_channel,dropout=0 , max_len=max_length)
        self.softmax = nn.Softmax(dim = -1)
        self.linear = nn.Linear(in_channel,in_channel)
        
    def forward(self,x):
        bs,c,h,w = x.shape
        k,v = x,x
        feature_maps = []
        # print(k.shape)
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            # print(k.shape)
            feature_maps.append(k)
        for i in range(len(self.k_decoder) -1):
            k = self.k_decoder[i](k)
            # print(k.shape)
            k = k + feature_maps[len(self.k_decoder)-i-2]
        k = self.k_decoder[-1](k)
        # print(k.shape)

        zeros = x.new_zeros((self.max_length,bs,c)) # (T,bs,c)
        q = self.position(zeros) 
        q = q.permute(1,0,2)
        q = self.linear(q)

        attn = torch.bmm(q,k.flatten(2,3))
        attn = attn / (c ** 0.5)
        attn = self.softmax(attn) # (bs,T,(h*c))

        v = v.permute(0,2,3,1).view(bs,-1,c)
        attn_vector = torch.bmm(attn,v)
        # print(attn_vector.shape)

        return attn_vector, attn

if __name__ == '__main__':
    sample = torch.zeros(32,2048,img_except_h//4,img_except_w//4)
    model = positionAttention(max_length=25)
    output,_ = model(sample)
    # print(output.shape)

