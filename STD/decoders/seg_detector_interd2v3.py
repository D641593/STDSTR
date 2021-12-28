from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d
import numpy as np
import random  
class InD2v3SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(InD2v3SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

#         self.out5 = nn.Sequential(
#             nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
#             #nn.Upsample(scale_factor=8, mode='nearest'))
        self.out5 = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)
        self.out4 = nn.Conv2d(inner_channels*2, inner_channels, 3, padding=1, bias=bias)
        self.out3 = nn.Conv2d(inner_channels*2, inner_channels, 3, padding=1, bias=bias)
        self.out2 = nn.Conv2d(inner_channels*2, inner_channels, 3, padding=1, bias=bias)
        
        self.down_conv2_1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 2, 1, 0, bias=False)
        )
        self.down_conv3_2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 2, 1, 0, bias=False)
        )
        self.down_conv4_3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 2, 1, 0, bias=False)
        )

        # self.M3d = nn.Sequential(
        #     nn.Conv2d(inner_channels, 2430, 4, stride=4, bias=bias),
        #     nn.PixelShuffle(3))
        
        # self.M2d = nn.Sequential(
        #     nn.Conv2d(inner_channels, 2430, 4, stride=4, bias=bias),
        #     nn.PixelShuffle(3))

        # self.M1d = nn.Sequential(
        #     nn.Conv2d(inner_channels, 2430, 4, stride=4, bias=bias),
        #     nn.PixelShuffle(3))

        self.M3u = nn.Sequential(
            nn.Conv2d(inner_channels, 2430, 2, stride=2, bias=bias),
            nn.PixelShuffle(3))
        
        self.M2u = nn.Sequential(
            nn.Conv2d(inner_channels, 2430, 2, stride=2, bias=bias),
            nn.PixelShuffle(3))

        self.M1u = nn.Sequential(
            nn.Conv2d(inner_channels, 2430, 2, stride=2, bias=bias),
            nn.PixelShuffle(3))

        

        self.M3 = nn.Sequential(
            nn.Conv2d(1038, inner_channels*2, 1, bias=bias),
            nn.Conv2d(inner_channels*2, inner_channels*2, 3, padding=1, bias=bias)
            )
            
        self.M2 = nn.Sequential(
            nn.Conv2d(1038, inner_channels*2, 1, bias=bias),
            nn.Conv2d(inner_channels*2, inner_channels*2, 3, padding=1, bias=bias)
            )
            
        self.M1 = nn.Sequential(
            nn.Conv2d(1038, inner_channels*2, 1, bias=bias),
            nn.Conv2d(inner_channels*2, inner_channels*2, 3, padding=1, bias=bias))
    
        self.M4 = nn.Sequential(
            nn.Conv2d(inner_channels*4, inner_channels*4, 1, bias=bias),
            nn.Conv2d(inner_channels*4, 29, 3, padding=1, bias=bias))
            
        self.M5 = nn.Sequential(
            nn.Conv2d(inner_channels*4, inner_channels*4, 1, bias=bias),
            nn.Conv2d(inner_channels*4, 29, 3, padding=1, bias=bias))
            
        self.M6 = nn.Sequential(
            nn.Conv2d(inner_channels*4, inner_channels*4, 1, bias=bias),
            nn.Conv2d(inner_channels*4, 28, 3, padding=1, bias=bias))
        
        self.U5 = nn.Sequential(
            nn.Conv2d(inner_channels+29, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 29, 3, padding=1, bias=bias)
            )
    
        self.U6 = nn.Sequential(
            nn.Conv2d(inner_channels+58, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 29, 3, padding=1, bias=bias)
           )
            
        self.U7 = nn.Sequential(
            nn.Conv2d(inner_channels+57, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 28, 3, padding=1, bias=bias)
            )
            
        self.U8 = nn.Sequential(
            nn.Conv2d(inner_channels+28, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 28, 3, padding=1, bias=bias)
            )
        
        ###down branch

        self.D1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels , 3, 
            padding=1,stride=2 ,bias=bias)
            )

        self.D2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels , 3, 
            padding=1,stride=2 ,bias=bias)
            )
        
        self.D3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 28, 3, padding=1, bias=bias)
            )

        self.D4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 1, bias=bias),
            nn.Conv2d(inner_channels, 28, 3, padding=1, bias=bias)
            )

        # self.M7 = nn.Sequential(
        #     nn.Conv2d(inner_channels*2, inner_channels, 1, bias=bias),
        #     nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias))

        # self.M8 = nn.Sequential(
        #     nn.Conv2d(inner_channels*3, inner_channels, 1, bias=bias),
        #     nn.Conv2d(inner_channels, 23, 3, padding=1, bias=bias))

        # self.M9 = nn.Sequential(
        #     nn.Conv2d(inner_channels*2, inner_channels, 1, bias=bias),
        #     nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias))

        # self.M10 = nn.Sequential(
        #     nn.Conv2d(inner_channels*3, inner_channels, 1, bias=bias),
        #     nn.Conv2d(inner_channels, 23, 3, padding=1, bias=bias))

         #------------------------------------------------

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)
        self.M1.apply(self.weights_init)
        self.M2.apply(self.weights_init)
        self.M3.apply(self.weights_init)
        self.M4.apply(self.weights_init)
        self.M5.apply(self.weights_init)
        self.M6.apply(self.weights_init)
        # self.M3d.apply(self.weights_init)
        # self.M2d.apply(self.weights_init)
        # self.M1d.apply(self.weights_init)
        self.M3u.apply(self.weights_init)
        self.M2u.apply(self.weights_init)
        self.M1u.apply(self.weights_init)
        self.down_conv2_1.apply(self.weights_init)
        self.down_conv3_2.apply(self.weights_init)
        self.down_conv4_3.apply(self.weights_init)
        # self.M7.apply(self.weights_init)
        # self.M8.apply(self.weights_init)
        # self.M9.apply(self.weights_init)
        # self.M10.apply(self.weights_init)
        self.D1.apply(self.weights_init)
        self.D2.apply(self.weights_init)
        self.D3.apply(self.weights_init)
        self.D4.apply(self.weights_init)
        self.U5.apply(self.weights_init)
        self.U6.apply(self.weights_init)
        self.U7.apply(self.weights_init)
        self.U8.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        
        u1 = self.out5(in5) 
        out4 = torch.cat([self.up5(u1),in4], dim=1)  # 1/16,512
        u2 = self.out4(out4)
        out3 = torch.cat([self.up4(u2),in3], dim=1)  # 1/8,512
        u3 = self.out3(out3)
        # print("u3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",c2.shape,c3.shape,c4.shape,c5.shape)
        out2 = torch.cat([self.up3(u3),in2], dim=1) # 1/4,512
        u4 = self.out2(out2)
#         print("out4,out3,out2",out4.shape,out3.shape,out2.shape)
#         print("u4,u3,u2,u1",u4.shape,u3.shape,u2.shape,u1.shape)
        
        #print("size---",c2.size()[2:][0],type(c2.size()[2:][0]))
        u4_u3_size = int((c2.size()[2:][0] + c3.size()[2:][0])/2) #120
        u3_u2_size = int((c3.size()[2:][0] + c4.size()[2:][0])/2) #60
        u2_u1_size = int((c4.size()[2:][0] + c5.size()[2:][0])/2) #30

        # print("middle size=======",u4_u3_size,u3_u2_size,u2_u1_size,type(u4_u3_size),type(u3_u2_size),type(u2_u1_size))

        # DownSample by 1chun
        tmp5_4 = self.down_conv2_1(u2) # 2*2 Conv
        tmp4_3 = self.down_conv3_2(u3)
        tmp3_2 = self.down_conv4_3(u4)        

        # print("tmp########################",tmp5_4.shape, tmp4_3.shape, tmp3_2.shape)
	
        tmp5_4_b, tmp5_4_c, tmp5_4_h, tmp5_4_w = tmp5_4.size() # 39
        tmp4_3_b, tmp4_3_c, tmp4_3_h, tmp4_3_w = tmp4_3.size() # 79
        tmp3_2_b, tmp3_2_c, tmp3_2_h, tmp3_2_w = tmp3_2.size() # 159

        # Index list(random choose middle layer's size to fit)
        index1 = np.linspace(0, tmp5_4_h, tmp5_4_h, endpoint=False, dtype=int)
        index2 = np.linspace(0, tmp4_3_h, tmp4_3_h, endpoint=False, dtype=int)
        index3 = np.linspace(0, tmp3_2_h, tmp3_2_h, endpoint=False, dtype=int)
        #print(out4_5_w, out3_4_w, out2_3_w)
        # print("random.........type------------",type(list(index1)),type(list(index1)[0]))
        index1_2 = np.array(random.sample(list(index1), u2_u1_size))
        index2_2 = np.array(random.sample(list(index2), u3_u2_size))
        index3_2 = np.array(random.sample(list(index3), u4_u3_size))
        
        index1_2 = torch.from_numpy(index1_2)
        index2_2 = torch.from_numpy(index2_2)
        index3_2 = torch.from_numpy(index3_2)
        
        # Row Index
        indexrow_sort1, _ = torch.sort(index1_2)
        indexrow_sort2, _ = torch.sort(index2_2)
        indexrow_sort3, _ = torch.sort(index3_2)
        
        index4 = np.linspace(0, tmp5_4_w, tmp5_4_w, endpoint=False, dtype=int)
        index5 = np.linspace(0, tmp4_3_w, tmp4_3_w, endpoint=False, dtype=int)
        index6 = np.linspace(0, tmp3_2_w, tmp3_2_w, endpoint=False, dtype=int)
        
        index1_3 = np.array(random.sample(list(index4), u2_u1_size))
        index2_3 = np.array(random.sample(list(index5), u3_u2_size))
        index3_3 = np.array(random.sample(list(index6), u4_u3_size))
        
        index1_3 = torch.from_numpy(index1_3)
        index2_3 = torch.from_numpy(index2_3)
        index3_3 = torch.from_numpy(index3_3)
        
        # Col Index
        indexcol_sort1, _ = torch.sort(index1_3)
        indexcol_sort2, _ = torch.sort(index2_3)
        indexcol_sort3, _ = torch.sort(index3_3)
        
        
        in6_row = tmp5_4[:, :,indexrow_sort1,:]
        in6 = in6_row[:, :, :,indexcol_sort1] #256,30
        # print("in6######################",in6.shape)
        # in6 = self.in8(in6)
        
        in7_row = tmp4_3[:, :,indexrow_sort2,:]
        in7 = in7_row[:, :, :,indexcol_sort2] #256,60
        # print("in7######################",in7.shape)
        # in7 = self.in7(in7)
        
        in8_row = tmp3_2[:, :,indexrow_sort3,:]
        in8 = in8_row[:, :, :,indexcol_sort3] #256,120
        #print("u3######################",u3.shape)
        # in8 = self.in6(in8)
        
        
        # h, w = p2.shape[2], p2.shape[3]
        
        # p6 = F.interpolate(in6, size=(h, w), mode='bilinear')
        # p7 = F.interpolate(in7, size=(h, w), mode='bilinear')
        # p8 = F.interpolate(in8, size=(h, w), mode='bilinear')
        
        # p6 = self.out6(p6)
        # p7 = self.out7(p7)
        # p8 = self.out8(p8)

        # fuse = torch.cat((p8, p7, p6, p5, p4, p3, p2), 1)
        ###############

        u4_d_c = in8 #256c #self.M3d(u4) 
        u3_u_c = self.M3u(u3) #270c
        u3_d_c = in7 #256c #self.M2d(u3)
        u2_u_c = self.M2u(u2) #270c
        u2_d_c = in6 #256c #self.M1d(u2)
        u1_u_c = self.M1u(u1) #270c
        # print("120x120--------",u4_d_c.shape,u3_u_c.shape)
        # print("60x60--------",u3_d_c.shape,u2_u_c.shape)
        # print("30x30--------",u2_d_c.shape,u1_u_c.shape)
        u4_d = F.interpolate(u4, size=int(u4_u3_size), mode='bilinear', align_corners= True) #
        u3_u = F.interpolate(u3, size=int(u4_u3_size), mode='bilinear', align_corners= True) #
        u3_d = F.interpolate(u3, size=int(u3_u2_size), mode='bilinear', align_corners= True) #
        u2_u = F.interpolate(u2, size=int(u3_u2_size), mode='bilinear', align_corners= True) #
        u2_d = F.interpolate(u2, size=int(u2_u1_size), mode='bilinear', align_corners= True) #
        u1_u = F.interpolate(u1, size=int(u2_u1_size), mode='bilinear', align_corners= True) #   
        # print("u4_d, u3_u,u4_d_c, u3_u_c--------------------------",u4_d.shape, u3_u.shape,u4_d_c.shape, u3_u_c.shape)
        m3 = self.M3(torch.cat([u4_d, u3_u,u4_d_c, u3_u_c], dim=1))
        m2 = self.M2(torch.cat([u3_d, u2_u,u3_d_c, u2_u_c], dim=1))
        m1 = self.M1(torch.cat([u2_d, u1_u,u2_d_c, u1_u_c], dim=1))
        
        m4 = self.M4(torch.cat([u4_d, u3_u,m3], dim=1))
        m5 = self.M5(torch.cat([u3_d, u2_u,m2], dim=1))
        m6 = self.M6(torch.cat([u2_d, u1_u,m1], dim=1))
        
        ####down branch
        d1 = self.D1(u1)
        d2 = self.D2(d1)

        u5 = self.U5(torch.cat([u4, F.interpolate(m4, size=c2.size()[2:][0], mode='bilinear', align_corners= True)], dim=1))
        
        u6 = self.U6(torch.cat([u3, F.interpolate(m4, size=c3.size()[2:][0], mode='bilinear', align_corners= True),
                                    F.interpolate(m5, size=c3.size()[2:][0], mode='bilinear', align_corners= True)], dim=1))     

        u7 = self.U7(torch.cat([u2, F.interpolate(m5, size=c4.size()[2:][0], mode='bilinear', align_corners= True),
                                    F.interpolate(m6, size=c4.size()[2:][0], mode='bilinear', align_corners= True)], dim=1))
        u8 = self.U8(torch.cat([u1, F.interpolate(m6, size=c5.size()[2:][0], mode='bilinear', align_corners= True)], dim=1))
        
        d4 = self.D4(d1)
        d3 = self.D3(d2)

        #####all resize to c2 size
        m4 = F.interpolate(m4, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        m5 = F.interpolate(m5, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        m6 = F.interpolate(m6, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        u6 = F.interpolate(u6, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        u7 = F.interpolate(u7, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        u8 = F.interpolate(u8, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        # m8 = F.interpolate(m8, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        # m10 = F.interpolate(m10, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        d3 = F.interpolate(d3, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        d4 = F.interpolate(d4, size=c2.size()[2:][0], mode='bilinear', align_corners= True)
        #print("all size-------",u5.shape,m4.shape,u6.shape,m5.shape,u7.shape,m6.shape,u8.shape)
                
        #fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = torch.cat((u5,m4,u6,m5,u7,m6,u8,d4,d3), 1)
        
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
