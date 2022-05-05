from email.mime import image
from pickletools import long1
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from skimage.io import imread
from skimage.transform import resize
import cv2

import json

with open('config.json') as jsonfile:
    data = json.load(jsonfile)

labels = []


img_path_name = 'imagenet-sample-images/n01443537_goldfish.jpg' #JPEG' n01443537_goldfish  n07715103_cauliflower  n04550184_wardrobe

image_label = img_path_name.split('/')[1]
image_label = image_label.split('.')[0]
image_label = image_label.split('_')[0] 

for i in data:
    if image_label == data[i][0]:
        print(data[i][1])
        ground_truth = int(i)
    labels.append(data[i][1])


class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.pretrained = models.alexnet(pretrained=True) #  alexnet resnet50 vgg16
        print('self.pretrained ', self.pretrained)
        layer_name = None
        #conv_layers = []

        for name, m in self.pretrained.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
                print(layer_name)

        #exit()
        #print(self.pretrained.features[28])
        #exit()

#        for name, m in self.pretrained.named_modules():
#            if name==layer_name:
#                self.layerhook.append(self.pretrained.register_forward_hook(se))

        print(layer_name)
        print(type(layer_name))
        self.layerhook.append(self.pretrained.features.register_forward_hook(self.forward_hook())) #[18][0]

        for name, param in self.pretrained.named_parameters():
            param.requires_grad = True
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        return out, self.selected_out

 #'Dataset//Dataset//truck.jpg' #'tiger.jfif' #'TestImage0.jpeg'
gcmodel = GradCamModel().to('cuda:0')
#img = imread('tiger.jfif') #'bulbul.jpg'
img = imread(img_path_name) #000000000785.jpg

img = resize(img, (224,224), preserve_range = True) #(224,224)
img = np.expand_dims(img.transpose((2,0,1)),0)
img /= 255.0
mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
img = (img - mean)/std
inpimg = torch.from_numpy(img).to('cuda:0', torch.float32)

out, acts = gcmodel(inpimg)

pred = out.argmax(dim=1)
print("Predicted label is ", labels[int(pred)], out[0][pred])
acts = acts.detach().cpu()

'''
loss = out[0][pred]
loss.backward()
#out.backward()
'''
label = torch.from_numpy(np.array([ground_truth])).to('cuda:0')#print('np.array([label]) ', np.array([label]))
loss = nn.CrossEntropyLoss()(out,label.type(torch.LongTensor).to('cuda:0'))
loss.backward()


grads = gcmodel.get_act_grads().detach().cpu()
pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()

for i in range(acts.shape[1]):
    acts[:,i,:,:] += pooled_grads[i]

heatmap_j = torch.mean(acts, dim = 1).squeeze()
heatmap_j_max = heatmap_j.max(axis = 0)[0]
heatmap_j /= heatmap_j_max
heatmap_j = resize(heatmap_j,(224,224),preserve_range=False)
img  = img*std + mean
img = np.transpose(img.squeeze(), (1, 2, 0))
remap_img = np.clip(img*255, 0, 255).astype('uint8')
remap_img = cv2.cvtColor(remap_img, cv2.COLOR_BGR2RGB)
heatmap_j = np.clip(heatmap_j*255, 0, 255).astype('uint8')
heatmap_j = cv2.applyColorMap(heatmap_j, cv2.COLORMAP_INFERNO) # cv2.COLORMAP_INFERNO)

overlay = cv2.addWeighted(remap_img, 0.3, heatmap_j, 0.7, 0.0)

cv2.imshow('heatmap_j', heatmap_j)
cv2.imshow('remap_img', remap_img)
cv2.imshow('overlay', overlay)
cv2.imwrite(img_path_name.split('.')[0] +  '_' + labels[pred] + '.jpg', overlay)
cv2.waitKey(-1)

for h in gcmodel.layerhook:
    h.remove()
for h in gcmodel.tensorhook:
    h.remove()