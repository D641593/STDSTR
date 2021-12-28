import torch
from model import *
from dataset import *
import torch.optim as optim
import logging
import os
from Levenshtein import distance as lvd
import yaml
import argparse

def train(yaml):
    # logging
    logging_fname = yaml['logging_file']
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO,filename=logging_fname,filemode='w',format=log_format,force = True)
    # device 
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    logging.info('train using %s'%deviceName)
    print('train using ',deviceName)
    # parameter for dataLoader
    batch_size = int(yaml['train']['batch_size'])
    max_length = int(yaml['train']['max_Length'])
    # data_loader
    data_root_dir = yaml['train']['dataset']['data_root_dir']
    trainLabelPath = yaml['train']['dataset']['train_label_file']
    charsDictPath = yaml['train']['dataset']['charsDict']
    dataset_STR = STRDataset(root=data_root_dir,labelPath=trainLabelPath,charsetPath=charsDictPath)
    data_root_dir = yaml['valid']['dataset']['data_root_dir']
    validLabelPath = yaml['valid']['dataset']['valid_label_file']
    charsDictPath = yaml['valid']['dataset']['charsDict']
    valid_dataset = STRDataset(root=data_root_dir,labelPath=validLabelPath,charsetPath=charsDictPath)
    valid_dataset.train = False
    data_loader = torch.utils.data.DataLoader(
        dataset_STR, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = collate_fn
    )
    # chars dictionary
    charsDict = dataset_STR.charsDict
    # model
    model = clsScore(len(charsDict.keys()),max_length)
    pretrain_model_path = str(yaml['train']['pretrain_model'])
    if pretrain_model_path != None and pretrain_model_path != 'None':
        model.load_state_dict(torch.load(pretrain_model_path))
    model.to(device)
    # optimizer
    lr = float(yaml['train']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CTCLoss(zero_infinity=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=2000)
    # set model to train mode
    model.train()
    # parameter for training and save
    step = 1
    epoches = int(yaml['train']['epoch'])
    log_step = int(yaml['train']['log_step'])
    best_valid = 0.0
    model_save_flag = False
    epoch_datanum = math.ceil(dataset_STR.__len__() / batch_size)
    model_save_root = yaml['train']['model_save_root']
    model_save_dir = yaml['train']['model_save_dir']
    if not os.path.exists(os.path.join(os.getcwd(),model_save_root,model_save_dir)):
        os.mkdir(os.path.join(os.getcwd(),model_save_root,model_save_dir))
    model_save_epoch = int(yaml['train']['model_save_epoch']) # save model every xxx epoch while training
    model_valid_epoch = int(yaml['train']['model_valid_epoch']) # valid model every xxx epoch while training
    # epoch for training
    for epoch in range(epoches):
        for imgs,labels in data_loader:
            # load img and label
            imgs = torch.stack(list(img.to(device) for img in imgs))
            labels = list(labels)
            labelIdx, labelLength = labels_IndexAndLength(labels = labels,charsDict=charsDict,max_Length = max_length-1)

            # pred and calculate ctc loss
            optimizer.zero_grad()
            outputs = model(imgs) # Batch_size / max_length / class_num(chars_num)
            outputs = outputs.permute(1,0,2) #  max_length / Batch_size / class_num(chars_num) for ctc loss
            outputLength = torch.full(size=(outputs.shape[1],),fill_value=max_length,dtype=torch.long)
            loss = loss_fn(outputs,labelIdx,outputLength,labelLength)
            if loss < 0:
                logging.info(labels)
            # step to step and writing log file
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if step % log_step == 0:
                logging.info("epoch: [%d] step: %d / %d, loss is %3.5f, lr : %3.5f"%(epoch,step%epoch_datanum,epoch_datanum,loss,optimizer.param_groups[0]["lr"]))
                # print("epoch: [%d] step: %d / %d, loss is %3.5f, lr : %3.5f"%(epoch,step%epoch_datanum,epoch_datanum,loss,optimizer.param_groups[0]["lr"]))
            step += 1
        
        # model valid
        if epoch%model_valid_epoch == 0:
            logging.info("epoch %d valid!"%(epoch))
            score = valid(data_loader_valid,model,device,charsDict)
            logging.info("1_NED : " + str(score))
            if score > best_valid:
                model_save_flag = True
            model.train()
        # model save
        if epoch % model_save_epoch == 0 or model_save_flag:
            model_save_name = "epoch_"+str(epoch)+".pth"
            path = os.path.join(model_save_root,model_save_dir,model_save_name)
            torch.save(model.state_dict(),path)
            logging.info("epoch %d save! model save at %s, loss is %f"%(epoch,path,loss))
            model_save_flag = False
    # model save after train
    model_save_name = 'final.pth'
    path = os.path.join(model_save_root,model_save_dir,model_save_name)
    torch.save(model.state_dict(),path)
    logging.info("That's it! model save at "+str(path))

def valid(dataloader,model,device,charsDict):
    model.eval()
    preds = []
    ans = []
    for imgs,labels in dataloader:
        imgs = torch.stack(list(img.to(device) for img in imgs))
        labels = list(labels)
        outputs = model(imgs) # Batch_size / max_length / class_num(chars_num)
        predIdx = torch.argmax(outputs,dim = 2)
        key_list = list(charsDict.keys())
        sentence = ""
        last_idx = 1
        for l in range(predIdx.shape[1]): # max_length
            pIdx = predIdx[0][l].cpu().detach().numpy()
            if pIdx == 1:
                break
            elif pIdx != 0 and pIdx != last_idx :
                sentence += key_list[predIdx[0][l]]
                last_idx = pIdx
            elif pIdx == 0:
                last_idx = 0
        preds.append(sentence)
        ans.append(labels[0])
    score = 0
    for i in range(len(ans)): # all data
        score += lvd(preds[i],ans[i]) / max(len(preds[i]),len(ans[i]))
    score = 1 - score / len(ans)
    return score

def labels_IndexAndLength(labels,charsDict,max_Length):
    global option_max_length
    global lost_word
    labelIdx = []
    labelLength = []
    for label in labels:
        if len(label) >= max_Length-1:
            label = label[:max_Length-1]
        for char in label:
            try:
                labelIdx.append(charsDict[char])
            except KeyError:
                if char not in lost_word:
                    lost_word.append(char)
                    logging.info("KeyError >> %c"%char)
                labelIdx.append(charsDict['@']) # @ mean don't care
        labelIdx.append(1) # append EOS symbol
        tmp = len(label)+1 # + EOS symbol
        if tmp > option_max_length:
            option_max_length = tmp
            print('option_max_length : %d / %s'%(option_max_length,label))
            logging.info('option_max_length : %d / %s'%(option_max_length,label))
        labelLength.append(tmp) 
    return torch.LongTensor(labelIdx), torch.LongTensor(labelLength)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file",default='train.yaml' ,help = 'path of the train yaml file', type = str)
    args = parser.parse_args()
    with open(str(args.yaml_file),'r',encoding='utf-8') as f:
        yaml = yaml.safe_load(f)
    option_max_length = 0
    lost_word = []
    train(yaml)
    with open('lost_word.txt','w',encoding = 'utf-8') as f:
        for char in lost_word:
            f.write(char+'\n')