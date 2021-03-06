import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import torch
import scipy.io as scio

class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        # print("self.data_list[0].....................",data_list)
        self.data_list = data_list or self.data_list
        
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    ### another datasets
    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                # print("is_training self.data_dir[i]===========================",self.data_dir[i],image_path)
                if 'TD500' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/train_txt/'+timg.strip().replace(".JPG","")+'.txt' for timg in image_list]
                elif 'ctw' in self.data_list[i] or 'logoHigh' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip().replace(".jpg","")+'.txt' for timg in image_list]
                elif 'mlt' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/train_gts/gt_'+timg.strip().replace(".jpg","").replace(".png","")+'.txt' for timg in image_list]
                
                else:
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                # print("is_valid self.data_dir[i]===========================",self.data_dir[i],image_path)
                if 'TD500' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_txt/'+timg.strip().replace(".JPG","")+'.txt' for timg in image_list]
                elif  'total_text' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                elif  'ctw' in self.data_list[i] or 'logoHigh' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip().replace(".jpg","")+'.txt' for timg in image_list]
                elif 'mlt' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/gt_'+timg.strip().replace(".jpg","").replace(".png","")+'.txt' for timg in image_list]
                else:
                    #gt_path=['/media/HDD/icdar2015/test_gts/gt_'+timg.strip().split('.')[0].replace(".jpg","")+'.txt' for timg in image_list]
                    #gt_path=[self.data_dir[i]+'/valid_gts/'+'gt_'+timg.strip().split('.')[0].replace(".jpg","")+'.txt' for timg in image_list]
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0].replace(".jpg","")+'.txt' for timg in image_list]
            # print(image_path)
            # print(gt_path)
            '''
            if self.is_training:
                # image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                image_path=[self.data_dir[i]+'/'+timg.strip() for timg in image_list]
                gt_path=['/media/HDD/small_syn/train/'+timg.strip().replace('jpg','txt') for timg in image_list]
                #gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                #image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                image_path=[self.data_dir[i]+'/'+timg.strip() for timg in image_list]
                print(self.data_dir[i])
                gt_path=['/media/HDD/small_syn/valid/'+timg.strip().replace('jpg','txt') for timg in image_list]
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]'''
                    
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
#         print(self.data_dir[0])
#         print('ctw' in self.data_dir[0])
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
#             print("gt-------------------",gt)
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                if label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'icdar' in self.data_dir[0] or 'mlt' in self.data_dir[0] or 'logoHigh' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'ctw' in self.data_dir[0]:
#                     print("ctw > poly............28",line[:28])
                    poly = np.array(list(map(float, line[:28]))).reshape((-1, 2)).tolist()
                    
                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)


class SynthDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    synthtext reading dataloader
    '''
    data_dir = State()
    processes = State(default=[])

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = self.data_dir[0]
        self.is_training = True
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []

        gt = scio.loadmat(self.data_dir + '/gt.mat')  # maybe data_dir[0]
        self.image_list = gt['imnames'][0]
        self.wordbox = gt['wordBB'][0]
        self.imgtxt = gt['txt'][0]
        for no, i in enumerate(self.imgtxt):
            all_words = []
            for j in i:
                all_words += [k.upper() for k in ' '.join(j.split('\n')).split() if k != '']
            self.imgtxt[no] = all_words

        self.get_all_samples()

    def get_all_samples(self):
        image_path = [self.data_dir + '/' + img[0] for img in self.image_list]
        self.image_paths += image_path

        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for box, txt in zip(self.wordbox, self.imgtxt):
            if box.ndim != 3:
                box = np.expand_dims(box, axis=2)
            box = box.transpose((2, 1, 0))  # num, 4, 2
            lines = []
            for i in range(len(txt)):
                item = {}
                item['poly'] = box[i]
                item['text'] = txt[i]
                #print("item================",item)
                lines.append(item)
            res.append(lines)
        return res


    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        #print("image_path",image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
