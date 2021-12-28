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
from model import Model
import matplotlib.pyplot as plt
# from thop import profile
# from ptflops import get_model_complexity_info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from skimage import io
import torch.nn.functional as F
#
#CUDA_VISIBLE_DEVICES=0 python demo_rec.py experiments/seg_detector/logo_resnet152_interd2v2.yaml --image_path=/media/HDD/icdar2015/test_images/ --resume ./weight/logo_interd2v2final_7142 --box_thresh 0.5 --visualize



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
    parser.add_argument('--box_thresh', type=float, default=0.5,
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

def warpPerspectivePoly(img, poly):
    '''print("polyyy",poly[0][0])
    poly= [[153.   3.]   poly[0]=polys [153.   3.] poly[0][0]=polyyy 153.0
 [396.   3.]   type=np.float32
 [396.  30.]
 [153.  30.]]
 '''
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k+1], poly[-k-2], poly[-k-1]])
        width += int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])/2))
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k+1], poly[-k-2], poly[-k-1]])
        #print("box===================",box) 
        '''box = [[1510.3125  500.625 ][1549.6875  500.625]
            [1549.6875  525.9375]
            [1510.3125  525.9375]]'''
        w = int(np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])/2)
        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32([[width_step,0],[width_step + w - 1,0],[width_step + w - 1,height-1]])
        #print("pts-----------------------------",pts1,pts2)
        '''
        pts1= [[1510.3125  500.625 ][1549.6875  500.625 ][1549.6875  525.9375]]
        pts2= [[ 0.  0.][58.  0.][58. 24.]]
        '''
        # img = img[0].numpy().transpose(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
        # print("img",type(img),img.shape)
        # print("w,h=============",width,height)
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)

        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask==1] = warped_img[warped_mask==1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32([[width_step,0],[width_step + w - 1,height-1],[width_step,height-1]])
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(warped_mask, (width_step,0), (width_step + w - 1,height-1), (0, 0, 0), 1)
        output_img[warped_mask==1] = warped_img[warped_mask==1]

        width_step += w
        
    return output_img

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

def recog(model,crop_path):
    # ############recognize!!!!!!!!!!!!!!!!
    converter = AttnLabelConverter(string.printable[:-6])
    if  os.path.isdir(crop_path):
        AlignCollate_demo = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False)
        demo_data = RawDataset(root=crop_path)  # use RawDataset folder name ='li'
        demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=192,
            shuffle=False,
            num_workers=4,
            collate_fn=AlignCollate_demo, pin_memory=True)
        
        # predict
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                recog_words = []
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([25] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, 26).fill_(0).to(device)
                
                stime = time.time() ##############################
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                
                preds_str = converter.decode(preds_index, length_for_pred)
                print("prcoessing time=",time.time()-stime)

                #log = open(f'./log_demo_result.txt', 'a')
                dashed_line = '-' * 80
                head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
                
                print(f'{dashed_line}\n{head}\n{dashed_line}')
                #log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    #print("`````````````",confidence_score)
                    #pred = pred.split()
                    img = cv2.imread(img_name)
                    #confs = confidence_score
                    if len(recog_words) ==0 :
                        recog_words = pred.split()
                        confs = str(confidence_score)[7:13].split()
                    else:
                        recog_words.append(pred.split()) 
                        # print("pred-----------------",pred)
                        confs.append(str(confidence_score)[7:13].split())
                    print("recog_words---------------",recog_words)
                    #cv2.imwrite('{}.jpg'.format(im_name), img)
                    #plt.imshow(img)
                    #plt.show()
                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
    return recog_words,confs
        #####################################
    
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

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        new_height = 1920
        new_width = 1920
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
        
    def format_output(self, batch, output, modelR):
        batch_boxes, batch_scores = output
        rec_words = []
        confs=[]
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            # print("filename",filename)
            img = io.imread(filename)
            # print("img................",img.shape,img.shape[0],img.shape[2])
            if img.shape[0] == 2: img = img[0]
            if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:   img = img[:,:,:3]
            img = np.array(img)

            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            image = batch['image']
            # print("boxxxxxx",boxes)
            scores = batch_scores[index]
            crop_path = os.path.join('./crop_bbox',result_file_name.replace(".txt","")) #./crop_bbox/79
            print("crop_path",crop_path)
            os.makedirs(crop_path, exist_ok=True)
            
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
                        # print("a............",boxes.shape,boxes.shape[0],boxes[i])
                        
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        
                        box = boxes[i,:,:].reshape(-1).tolist()
                        # print("boxaaaaaaaaaaaaa",box)
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + "\n")
                        
                        ##########recog!!!!!!!
                        rect_img = warpPerspectivePoly(img , boxes[i])
                        # print("rect_img",rect_img.shape)
                        rect_img_path = crop_path +'/'+ str(i) + '.jpg'
                        cv2.imwrite((rect_img_path),rect_img)
                        rec_words,confs = recog(modelR,crop_path)
                        ##########recog!!!!!!!
        return rec_words,confs
   
    
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
        rec_words = []
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

            ###recog!!!!!!!!!!!!!
            modelR = Model()
            modelR = torch.nn.DataParallel(modelR).to(device)
            modelR.load_state_dict(torch.load('TPS-ResNet-BiLSTM-Attn.pth', map_location=device))
            ###recog!!!!!!!!!!!

            # print(gray.shape)
            # print("gray=",gray)
            # print("path===",os.path.join(self.args['result_dir'],'pred', image_path.split('/')[-1].split('.')[0]+'.jpg'))
            cv2.imwrite(os.path.join(self.args['result_dir'],'pred', image_path.split('/')[-1].split('.')[0]+'.jpg'), gray*255)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            rec_words,confs = self.format_output(batch, output,modelR)
            

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output, rec_words,confs)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

if __name__ == '__main__':
    main()
