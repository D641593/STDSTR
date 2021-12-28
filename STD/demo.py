#!python3
import argparse
import os
import string
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import time
##recog part
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
#from model import Model
import matplotlib.pyplot as plt
# from thop import profile
# from ptflops import get_model_complexity_info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import file_utils
##CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path img_10.jpg --resume totaltext_resnet18 --polygon --box_thresh 0.7 --visualize
#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --image_path=./test --resume totaltext_resnet18 --polygon --box_thresh 0.7 --visualize
#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/syn_resnet50_deform_thre.yaml --image_path=/media/HDD/icdar2015/test_images --resume ./workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final --polygon --box_thresh 0.7 --visualize
#
#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml --image_path=./test --resume resnet50_M.pth --box_thresh 0.7 --visualize
#
#CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/ic15_resnet152_interd2v2.yaml --image_path=/media/HDD/icdar2015/test_images/img_10.jpg --resume /media/HDD/下載/model_epoch_829_minibatch_34000.pth --box_thresh 0.55

###backbone deformable_resnet50

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    #parser.add_argument('--test_folder', default='./test/', type=str, help='folder path to input images')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=1152,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    
    
    t = time.time()
    # load data
    
    
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    """ For test images in a folder """
    if os.path.isdir(args['image_path']):
        print("loading folder:",args['image_path'])
        image_list, _, _ = get_files(args['image_path'])
        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
            Demo(experiment, experiment_args, cmd=args).inference(image_path, args['visualize'])
    else:
        print("loading image:",args['image_path'])
        Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])
    
    '''for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])
        
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)'''

    print("elapsed time : {}s".format(time.time() - t))
    
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files
    
class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img, fix_height = 1920, fix_width = 1920):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        new_height = fix_height
        new_width = fix_width
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            # print("boxxxxxx",boxes)
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        # print("a............",box)
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        #print("a............",boxes.shape,boxes.shape[0])
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + "\n")
   
    
    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        #all_matircs = {}
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            # flops, params = profile(model, inputs=(1, 3, 256,256))
            # flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
            #                                print_per_layer_stat=True, verbose=True)
  
            # print("flops................",flops,params)
            gray = pred.squeeze(1)
            # print(pred.shape)
            gray = pred.cpu().detach().numpy().squeeze()
            gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            # print(gray.shape)
            # print("gray=",gray)
            print("path===",os.path.join(self.args['result_dir'],'pred', image_path.split('/')[-1].split('.')[0]+'.jpg'))
            cv2.imwrite(os.path.join(self.args['result_dir'],'pred', image_path.split('/')[-1].split('.')[0]+'.jpg'), gray*255)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

if __name__ == '__main__':
    main()
