import torch
import torch.distributed as dist
from tqdm import tqdm


def train(net,trainloader,optimizer,criterion,device,epoch,scheduler,run):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs1, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()
    acc=100*correct/total
    
    run["train_acc"].log(acc)
    run["train_loss"].log(running_loss/len(trainloader))
    print('[%d] loss: %.3f' % (epoch + 1, running_loss /len(trainloader)))
    print(len(trainloader))
    
    return acc

def test(net,testloader,criterion,device,run):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader,0)):
            images1, labels = data[0].to(device), data[1].to(device)
            outputs = net(images1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)

            running_loss += loss.item()
    acc=100*correct/total
    run["test_acc"].log(acc)
    run["test_loss"].log(running_loss/len(testloader))
    print('Accuracy of the network on the test images: %d %%' %(acc))
    print(correct , total)