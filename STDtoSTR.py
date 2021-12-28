import cv2
import os
import numpy as np
import csv
import math
import argparse

def getTargetPoints(points):
    dis = []
    for i in range(4):
      p = points[i]
      nextp = points[(i+1)%4]
      dis.append(math.hypot(p[0]-nextp[0],p[1]-nextp[1]))
    w = max(dis[0],dis[2])
    h = max(dis[1],dis[3])
    return np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float32),(int(w),int(h))

def warpImg(img,points,scale = 1):
    points = np.array(points,dtype=np.float32) * scale # for rescale
    points = points.reshape(4,2)
    targets,shape = getTargetPoints(points)
    M = cv2.getPerspectiveTransform(points,targets)
    transImg = cv2.warpPerspective(img,M,shape,cv2.INTER_LINEAR)
    return transImg,points

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--res_img_dir", default='STD/demo_results/',help = 'path of the detection result file', type = str)
  parser.add_argument("--img_dir",default = 'STD/datasets/logoHigh/private', help = 'path of image dir', type = str)
  parser.add_argument("--output_dir",default = "STR/train_data/private_high_crop", help = 'output dir of crop image', type = str)
  args = parser.parse_args()
  rootdir = args.res_img_dir
  imgdir = args.img_dir
  outdir = args.output_dir
  os.rmdir(outdir)
  os.mkdir(outdir)

  txtfiles = [i for i in sorted(list(os.listdir(rootdir))) if '.txt' in i]
  # txt_content = []
  with open('STR/empty.csv','w',encoding='utf-8') as wf:
    writer = csv.writer(wf)
    for fname in txtfiles:
      with open(os.path.join(rootdir,fname),'r',encoding='utf-8') as f:
          txt = f.readlines()
      img = cv2.imread(os.path.join(imgdir,fname[4:-3]+'jpg'))
      index = 1
      for t in txt:
        csv_row = [fname[4:-4]]
        t = t.split(',')
        transImg,points = warpImg(img,t,scale = 1)
        h,w = transImg.shape[:2]
        if h <= 10 or w <= 10:
          continue
        csv_row.extend(list(points.reshape(-1).astype('int32')))
        writer.writerow(csv_row)
        h,w = transImg.shape[:2]
        if h > w:
            transImg = cv2.rotate(transImg,cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        transImgfname = fname[4:-4] + "_" + str(index) + ".jpg"
        index += 1
        cv2.imwrite(os.path.join(outdir,transImgfname),transImg)
        # row = os.path.join(outdir,transImgfname) + '\n'
        # txt_content.append(row)
      print(fname+' finish!')
