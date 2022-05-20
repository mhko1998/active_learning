import torch
import torch.nn as nn
import os
import timm

import ast
import dataset
import module

def activerun(method, save_path, teacher_path, startpoint, max_epoch, dataset_path, batch_size, num_classes, ratio, run, teacher):
    device = 'cuda:0'
    
    beforeimages = []
    
    if teacher != 'base' and teacher != '':
        list = open(teacher_path+"trainset.txt",'r')
        line = list.readline()
        beforeimages = ast.literal_eval(line)
    
    trainloader, testloader, beforeimages = dataset.data_loader(ratio, beforeimages, method, batch_size, teacher_path, startpoint, dataset_path)
    
    net = module.net(num_classes)
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer=torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,75,90],gamma=0.1)
    saver = timm.utils.CheckpointSaver(net, optimizer, checkpoint_dir=save_path, max_history = 2)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    name_of_file = "trainset"
    completename = os.path.join(save_path, name_of_file+".txt")
    file1 = open(completename,"w")
    file1.write(str(beforeimages))
    file1.close

    for epoch in range(max_epoch):
        acc = module.train(net,trainloader,optimizer,criterion,device,epoch,scheduler,run)

        if epoch % 5 == 4:
            module.test(net,testloader,criterion,device,run)
        saver.save_checkpoint(epoch, metric = acc)
    
    print('Finished Training')