import os
from pyexpat import model
from re import M
import torch
import time
import pickle
import argparse
import torch.nn as nn
from Decoder import RNN
from utils import get_cnn
import matplotlib.pyplot as plt
from Vocabulary import Vocabulary
from torchvision import transforms
from torch.autograd import Variable
from Preprocess import load_captions
from DataLoader import DataLoader, shuffle_data
import numpy as np
import cv2
from skimage.transform import resize

class GradCamModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.model = get_model()
        self.cnn, self.lstm = self.model[0], self.model[1]
        self.layerhook.append(self.cnn.alexnet.features.register_forward_hook(self.forward_hook()))

        for name, param in self.cnn.named_parameters():
            param.requires_grad = True

        for name, param in self.lstm.named_parameters():
            param.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def zero_grad(self):
        self.cnn.zero_grad()
        self.lstm.zero_grad()
        return

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, image, caption):

        cnn_out = self.cnn(image)
        lstm_out = self.lstm(cnn_out, caption)

        return lstm_out, self.selected_out


def calc_grad_cam_rgb(grads, acts, img):

    pooled_grads = torch.mean(grads, dim=[0,2,3])

    mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))

    for i in range(acts.shape[1]):
        acts[:,i,:,:] += pooled_grads[i]

    heatmap_j = torch.mean(acts, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
    heatmap_j = heatmap_j.numpy()
    heatmap_j = resize(heatmap_j,(224,224),preserve_range=True)
    img  = img*std + mean
    img = np.transpose(img.squeeze(), (1, 2, 0))
    remap_img = np.clip(img*255, 0, 255).astype('uint8')
    remap_img = cv2.cvtColor(remap_img, cv2.COLOR_BGR2RGB)
    heatmap_j = np.clip(heatmap_j*255, 0, 255).astype('uint8')
    heatmap_j = cv2.applyColorMap(heatmap_j, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(remap_img, 0.7, heatmap_j, 0.3, 0.0)

    cv2.imshow('heatmap_j', heatmap_j)
    cv2.imshow('remap_img', remap_img)
    cv2.imshow('overlay', overlay)
    cv2.waitKey(-1)

def get_model():

    vocab_size = vocab.index

    model_name = args.model
    cnn = get_cnn(architecture = model_name, embedding_dim = args.embedding_dim)
    lstm = RNN(embedding_dim = args.embedding_dim, hidden_dim = args.hidden_dim, 
                vocab_size = vocab_size)

    iteration = 200
    model_name = args.model
    cnn_file = 'iter_' + str(iteration) + '_cnn.pkl'
    lstm_file = 'iter_' + str(iteration) + '_lstm.pkl'
    cnn.load_state_dict(torch.load(os.path.join(model_name, cnn_file), map_location=torch.device('cpu')))
    lstm.load_state_dict(torch.load(os.path.join(model_name, lstm_file),map_location=torch.device('cpu')))

    return cnn, lstm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model = "alexnet"
    args.dir = "test"
    args.embedding_dim = 512
    args.hidden_dim = 512
    train_dir = args.dir
    threshold = 5

    with open(os.path.join(args.model, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))
                                    ])

    dataloader = DataLoader(train_dir, vocab, transform)
    data = dataloader.gen_data()
    print(train_dir + ' loaded')

    gcmodel = GradCamModel()
    criterion = nn.CrossEntropyLoss()
    
    shuffled_images, shuffled_captions = shuffle_data(data, seed = 1)
    num_captions = len(shuffled_captions)
    loss_list = []
    tic = time.time()

    for i in range(num_captions):
        
        image_id = shuffled_images[i]
        print('image_id: ',image_id)

        image = dataloader.get_image(image_id)
        image = image.unsqueeze(0)
                    
        image = Variable(image)
        caption = torch.LongTensor(shuffled_captions[i])

        caption_train = caption[:-1] # remove <end>
        gcmodel.zero_grad()

        lstm_out, acts = gcmodel(image, caption_train)
        loss = criterion(lstm_out, caption)
        loss.backward()

        grads = gcmodel.get_act_grads().detach().cpu()
        acts = acts.detach().cpu()
        image = image.detach().cpu().numpy()

        print('loss: ', loss.item())            
        calc_grad_cam_rgb(grads, acts, image)
        

