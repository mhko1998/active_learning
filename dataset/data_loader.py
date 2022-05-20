import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import glob
import random
import ast
import os
import torchvision
from tqdm import tqdm
import torch.nn as nn

import argparse
import utils

class ImageDataLoader(Dataset):
    def __init__(self, dir ,images, transform):
        self.images = images
        self.transform = transform
        self.dir = dir
        self.label_dict, self.name_dict=self.__labeling__()
        
    def __labeling__(self):
        dirname = self.dir
        label_dict = dict()
        name_dict = dict()
        i= [i for i in range(len(dirname))]
        for y in range(len(dirname)):
            label = dirname[y].split('/')[-1]
            label_dict[label] = i[y]
            name_dict[i[y]]=label
        return label_dict, name_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self ,index):
        imgname = self.images[index]
        x=imgname.split('/')[-2]
        label=self.label_dict[x]
        image = Image.open(imgname)
        image=self.transform(image)
        return image, label

def data_loader(ratio, beforeimages, method, batch_size, teacher_path, startpoint, dataset_path):
    beforeimages=[]
    traintrans=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
    testtrans=transforms.Compose([transforms.ToTensor(),])
    trainimages= glob.glob(dataset_path+'/train/*/*.png')
    traindir = glob.glob(dataset_path+'/train/*')

    if method=="random":
        trainimages, beforeimages= utils.random_reduce(ratio, trainimages, beforeimages)
    
    elif method=="leconf_reduce":
        trainimages, beforeimages= utils.leconf_reduce(ratio, trainimages, beforeimages, traindir, testtrans, batch_size, teacher_path, startpoint)
    
    testimages=glob.glob(dataset_path+'/test/*/*.png')
    testdir = glob.glob(dataset_path+'/test/*')
    
    trainset=ImageDataLoader(traindir,trainimages,traintrans)
    testset=ImageDataLoader(testdir,testimages,testtrans)
    
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=8)
    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=8)
    
    return trainloader, testloader, beforeimages