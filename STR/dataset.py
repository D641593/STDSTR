import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import re

img_except_h, img_except_w = 32,512 # 32,512
class STRDataset(torch.utils.data.Dataset):
    def __init__(self, root, labelPath, charsetPath,train=True):
        self.root = root
        self.labelPath = labelPath
        self.charsetPath = charsetPath
        self.imgPaths = []
        self.labels = []
        self.charsDict = {}
        self.train = train
        with open(os.path.join(self.root, self.labelPath),'r',encoding = 'utf-8') as f:
            txt = f.readlines()
            for row in txt:
                row = row.split('\t')
                path = row[0]
                label = row[1][:-1] # except '\n'
                self.imgPaths.append(os.path.join(self.root,path))
                self.labels.append(label)
        self.imgPaths, self.labels = (list(i) for i in zip(*sorted(zip(self.imgPaths,self.labels))))

        with open(os.path.join(self.root,self.charsetPath),'r',encoding='utf-8') as f:
            txt = f.readlines()
            self.charsDict['blank'] = 0
            self.charsDict['EOS'] = 1
            idx = 2
            for char in txt:
                self.charsDict[char[0]] = idx # char[0] to expect '\n'
                idx += 1
            self.charsDict['@'] = len(self.charsDict)

    def __getitem__(self, idx):
        img = Image.open(self.imgPaths[idx])
        label = self.labels[idx]
        if self.train:
            trans = get_train_transforms(img.height,img.width)
        else:
            trans = get_eval_transforms(img.height,img.width)
        img = trans(img)
        w = img.shape[2]
        half = (img_except_w - w) // 2
        offset = (img_except_w - w) % 2
        p1d = (half,half+offset)
        img = nn.functional.pad(img,p1d,'constant',0)
        return img,label

    def __len__(self):
        return len(self.imgPaths)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transforms(h,w):
    ratio = img_except_h / h
    ratio_w = int(ratio * w)
    if ratio_w > img_except_w:
        ratio_w = img_except_w
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_except_h,ratio_w)),
        transforms.RandomRotation((-10,10)),
        transforms.RandomCrop((img_except_h,ratio_w),pad_if_needed=True),
        transforms.RandomPerspective(distortion_scale=0.5, p = 0.3)
    ])

def get_eval_transforms(h,w):
    ratio = img_except_h / h
    ratio_w = int(ratio * w)
    if ratio_w > img_except_w:
        ratio_w = img_except_w
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_except_h,ratio_w))
    ])

