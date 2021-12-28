import logging
import torch
import os
from dataset import *
from model import *
from PIL import Image
import math
import argparse

img_except_h, img_except_w = 32,512 # 32,512

def get_charsDict_and_model(dict_path, model_path):
    # device
    # deviceName = 'cpu'
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    print('demo using ',deviceName)
    # Dict
    charsDict = {} 
    with open(os.path.join(dict_path),'r',encoding='utf-8') as f:
        txt = f.readlines()
        charsDict['blank'] = 0
        charsDict['EOS'] = 1
        idx = 2
        for char in txt:
            charsDict[char[0]] = idx # char[0] to expect '\n'
            idx += 1
    charsDict['@'] = len(charsDict)
    # model 
    max_length = 40+1
    model = clsScore(len(charsDict.keys()),max_length)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return charsDict,model

def demo(img = None,image_fname = None,model = None,charsDict = None):
    assert img != None or image_fname != None,'please input the PIL image or image_file_path'
    assert model != None, 'please offer a model to demo'
    assert charsDict != None, "please offer the charsDict of demo model"
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    # image
    with torch.no_grad():
        if img == None and image_fname != None:
            img = Image.open(image_fname)
        trans = get_eval_transforms(img.height,img.width)
        img = trans(img)
        img = torch.unsqueeze(img,dim=0) # batch_size = 1
        w = img.shape[3]
        half = (img_except_w - w) // 2
        offset = (img_except_w - w) % 2
        p1d = (half,half+offset)
        img = nn.functional.pad(img,p1d,'constant',0)
        img = img.to(device)
        # predict
        pred = model(img)
        predIdx = torch.argmax(pred,dim=2)
        predconfs = torch.max(pred,dim=2).values

        key_list = list(charsDict.keys())
        sentence = ""
        confs = 0
        last_idx = 1
        for l in range(predIdx.shape[1]): # max_length
            pIdx = predIdx[0][l].cpu().detach().numpy()
            if pIdx == 1:
                break
            elif pIdx != 0 and pIdx != last_idx :
                sentence += key_list[predIdx[0][l]]
                confs += math.exp(predconfs[0][l].cpu().detach().numpy())
                last_idx = pIdx
            elif pIdx == 0:
                last_idx = 0

        if len(sentence) != 0:
            confs = confs / len(sentence)
        return sentence, confs