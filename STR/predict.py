from PIL import Image
from demo import *
import logging
import os
import csv
from paddleocr import PaddleOCR
import cv2
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file",default='predict.yaml', help = 'path of the train yaml file', type = str)
    args = parser.parse_args()
    with open(str(args.yaml_file),'r',encoding='utf-8') as f:
        yaml = yaml.safe_load(f)
    # model and charsDict
    charsDict,model = get_charsDict_and_model(dict_path = str(yaml['charsDict']), model_path = str(yaml['model_path']))
    # predict
    imgdir = yaml['image_dir']
    if type(imgdir) is not list:
        imgdir = [imgdir]
    except_list = ['','@']
    with open(str(yaml['input_csv']),'r',encoding='utf-8') as f:
        rows = csv.reader(f)
        with open(str(yaml['output_csv']),'w',encoding='utf-8') as wf:
            writer = csv.writer(wf)
            last_fname = ""
            index = 1
            for row in rows:
                csv_row = row
                if last_fname == row[0]:
                    index += 1
                else:
                    last_fname = row[0]
                    index = 1
                fname = row[0]+"_"+str(index)+".jpg"
                # print(index)
                option = []
                optionconf = []
                for dirpath in imgdir:
                    imgfname = os.path.join(dirpath, fname)
                    img = Image.open(imgfname)
                    paddleImg = cv2.imread(imgfname)
                    sentence = ""
                    if img.width / img.height > 10:
                        top = 0
                        bottom = img.height
                        cropImgs = []
                        splitnum = int((img.width / img.height) / 5)
                        for i in range(splitnum):
                            left = 0 + int(img.width / splitnum * i)
                            right = int(img.width / splitnum * (i + 1))
                            cropImgs.append(img.crop((left,top,right,bottom)))
                        for cropImg in cropImgs:
                            pred,_ = demo(img = cropImg,model = model,charsDict = charsDict)  
                            sentence += pred
                    else:
                        sentence,_ = demo(img = img,model = model, charsDict = charsDict)
                
                flag = 1
                tmp = ''
                for i in range(len(sentence)):
                    if sentence[i] == '@' and flag == 1:
                        tmp = tmp + "@"
                        flag = 0
                    else:
                        tmp = tmp + sentence[i]
                        flag = 1
                
                sentence = tmp
                if sentence not in except_list: # this is useful
                    sentence = sentence.replace(' ','')
                    sentence = sentence.replace("@","#")
                    csv_row.append(sentence)
                    writer.writerow(csv_row)
        