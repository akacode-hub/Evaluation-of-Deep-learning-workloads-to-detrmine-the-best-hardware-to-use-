import argparse
#from utilities import text_helper
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models import VqaModel
from skimage.transform import resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.model = get_model()
        self.layerhook.append(self.model.img_encoder.model.features.register_forward_hook(self.forward_hook()))

        for name, param in self.model.named_parameters():
            param.requires_grad = True
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            print('out shape: ', out.shape)
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, img, quest):
        out = self.model(img, quest)
        return out, self.selected_out

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def get_dataloader():

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='valid.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    return data_loader

def get_model():

    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)

    loaded_model = torch.load(args.saved_model)['state_dict']
    model.load_state_dict(loaded_model)
    #model.eval()

    return model

def predict(data_loader):

    model = get_model()
    phase = 'train'

    for batch_idx, batch_sample in enumerate(data_loader[phase]):

        image = batch_sample['image'].to(device)
        question = batch_sample['question'].to(device)
        label = batch_sample['answer_label'].to(device)

        output = model(image, question)
    
        predicts = torch.softmax(output, 1)
        probs, indices = torch.topk(predicts, k=5, dim=1)
        probs = probs.squeeze()
        indices = indices.squeeze()
        print("predicted - probabilty")

        for i in range(5):
            print("'{}' - {:.4f}".format(ans_vocab[indices[i].item()], probs[i].item()))


def grad_cam(data_loader):

    phase = 'train'
    model = GradCamModel().to('cuda:0')

    for batch_idx, batch_sample in enumerate(data_loader[phase]):

        image = batch_sample['image'].to(device)
        question = batch_sample['question'].to(device)
        label = batch_sample['answer_label'].to(device)

        with torch.set_grad_enabled(phase == 'train'):
            output, acts = model(image, question)      # [batch_size, ans_vocab_size=1000]
            loss = criterion(output, label)
            loss.backward()    

            grads = model.get_act_grads().detach().cpu()
            acts = acts.detach().cpu()
            image = image.detach().cpu().numpy()
            calc_grad_cam_rgb(grads, acts, image)

        if batch_idx > 1: break


def calc_grad_cam_rgb(grads, acts, img):

    pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()

    mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))

    for i in range(acts.shape[1]):
        acts[:,i,:,:] += pooled_grads[i]

    heatmap_j = torch.mean(acts, dim = 1).squeeze()
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.input_dir = './datasets'
    args.max_qst_length = 30
    args.max_num_ans = 10
    args.batch_size = 1
    args.num_workers = 1
    args.embed_size = 1024
    args.word_embed_size = 300
    args.num_layers = 2
    args.hidden_size = 512
    args.saved_model = "./models/model-epoch-30.ckpt"    

    data_loader = get_dataloader()
    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    qst_vocab = load_str_list("datasets/vocab_questions.txt")
    ans_vocab = load_str_list("./datasets/vocab_answers.txt")

    criterion = nn.CrossEntropyLoss()

    # predict(data_loader)
    grad_cam(data_loader)