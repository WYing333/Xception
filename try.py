""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import numpy as np

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(3)

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x



class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2): #num_classes=1000  #改成1 靠近哪个 就是哪个 原本是2
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


#GPU
#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
#print(device)
model = xception().cuda(3)#to(device) #******#


#loss
#m = nn.Sigmoid() #0-1
loss_fn = nn.BCELoss()

#x = torch.randn(4, 2)  # (4, 2)
#target = torch.empty(4).random_(2)  # shape=(4, ) 其中每个元素值为0或1
#onehot_target=torch.eye(2)[target.long(), :]  # (4, 2)

# print(x)
# print(m(x))
# print(onehot_target)

#loss = loss_fn(m(x), onehot_target)
#print('Total loss for this batch: {}'.format(loss.item()))
#print(loss)


#optimizer
optimizer = torch.optim.Adam(model.parameters()) #, lr=LR


#train loop

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torch.utils.data import DataLoader
from CustomDataset import CustomDataset

from torchvision.transforms import transforms

from sklearn.metrics import roc_auc_score


train_filename="/home/ywang/train_balanced_label.txt"
val_filename="/home/ywang/validation_label.txt"
#test_filename="/home/ywang/test_label.txt"
image_dir='/home/ywang/XceptionExtract'
 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)])
 

batch_size=32  #训练时的一组数据的大小

train_data = CustomDataset(annotations_file=train_filename, image_dir=image_dir,repeat=1, transform=transform)
val_data = CustomDataset(annotations_file=val_filename, image_dir=image_dir,repeat=1, transform=transform)
#test_data = CustomDataset(filename=test_filename, image_dir=image_dir,repeat=1, transform=transform)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.cuda(3)#to(device)#******#
        labels = labels[0]
        labels = labels.cuda(3)#to(device)#******#

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
  
        m = nn.Sigmoid() #0-1

        onehot_labels = torch.eye(2)[labels.long(), :]  #eye(2)
        onehot_labels = onehot_labels.cuda(3)#to(device)
        loss = loss_fn(m(outputs), onehot_labels)
        #labels = labels.unsqueeze(1) #target: (torch.Size([64])) to (torch.Size([64, 1])) 
        #labels = labels.float()
        #loss = loss_fn(m(outputs), labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    #model.train()
    #avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.eval()

    running_vloss = 0.0
    voutputs_all = []
    label_all = []

    for i, vdata in enumerate(val_dataloader):
        vinputs, vlabels = vdata
        vinputs = vinputs.cuda(3)#to(device)#******#
        vlabels = vlabels[0]
        vlabels = vlabels.cuda(3)#.to(device)#******#

        voutputs = model(vinputs)
        #voutputs = voutputs.cuda(3)#.to(device)#******#
        voutputs_all.extend(voutputs.cpu().detach().numpy()) #[:,1] .detach()
        label_all.extend(vlabels)
        
        for i in range(0,len(label_all)):
            label_all[i] = label_all[i].long()
        onehot_vlabels = torch.eye(2)[label_all, :]

        voutputs_alls = torch.Tensor(np.array(voutputs_all))

        m = nn.Sigmoid() #0-1
        #onehot_vlabels = torch.eye(1)[vlabels.long(), :] #eye(1)
        #onehot_vlabels = onehot_vlabels.to(device)
        #vloss = loss_fn(m(voutputs), onehot_vlabels)
        #vlabels = vlabels.unsqueeze(1)
        #vlabels = vlabels.float()
        #print(onehot_vlabels)
        #vloss1 = loss_fn(m(voutputs),vlabels)
        vloss = loss_fn(m(voutputs_alls), onehot_vlabels)

        #running_vloss1 += vloss1
        running_vloss += vloss

    #avg_vloss1 = running_vloss1 / (i + 1)
    avg_vloss = running_vloss / (i + 1)
    #print('LOSS valid {}'.format(avg_vloss1))
    print('LOSS valid {}'.format(avg_vloss))
    #print("AUC val:{:.4f}".format(roc_auc_score(vlabels,m(voutputs))))   
    print("AUC val:{:.4f}".format(roc_auc_score(onehot_vlabels, m(voutputs_alls))))  

    # Log the running loss averaged per batch
    # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                 epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    # if avg_vloss < best_vloss:
    #     best_vloss = avg_vloss
    #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    #     torch.save(model.state_dict(), model_path)

    epoch_number += 1


#saved_model = Xception()
#saved_model.load_state_dict(torch.load(PATH))