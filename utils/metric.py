import json
import random
import torch
import os
from tqdm import tqdm

import dataset
import module

def read_conf(json_path):
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config

def random_reduce(ratio, trainimages, beforeimages):
    originlen=len(trainimages)

    for i in beforeimages:
        trainimages.remove(i)
   
    rd_trainimages = random.sample(trainimages, k=int(ratio*originlen))
    rd_trainimages.extend(beforeimages)
    
    return rd_trainimages

def leconf_reduce(ratio, trainimages, beforeimages, traindir, testtrans, batch_size, teacher_path, startpoint, num_classes):
    originlen=len(trainimages)
    
    for i in beforeimages:
        trainimages.remove(i)
    
    check_cf_testset=dataset.ImageDataLoader(traindir, trainimages, testtrans)
    check_cf_testloader=torch.utils.data.DataLoader(check_cf_testset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    device = 'cuda:0'
    net = module.net(num_classes)
    net.to(device)

    net.load_state_dict(torch.load(teacher_path+'/last.pth.tar', map_location= device)['state_dict'])

    net.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(check_cf_testloader,0)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            outputs = torch.softmax(outputs, dim=1)
            predicted, _= torch.max(outputs.data, 1)
            
            if i==0:
                a=predicted
            else:
                a=torch.cat((a,predicted),0)
    
    sorted, indices = torch.sort(a,0)
    indices=indices[int(ratio*originlen)*(0+startpoint):int(ratio*originlen)*(1+startpoint)]
    
    rd_trainimages=[]
    for i in indices:
        rd_trainimages.append(trainimages[i])
    rd_trainimages.extend(beforeimages)
    return rd_trainimages