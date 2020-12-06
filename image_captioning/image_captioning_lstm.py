# Imports

import random 
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
import re 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt 
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# Loading data

features = np.load("test_features.npy",allow_pickle=True) # images
unique_words = np.load("words.npy",allow_pickle=True) # vocabulary
initial_embeddings = np.load("initial_embeddings_100.npy",allow_pickle=True) # initial glove embeddings 
all_captions = np.load("test_references.npy",allow_pickle=True) # All 5 captions for all images
captions = all_captions
unique_words = unique_words.tolist()

# NetVLAD CNN Encoder Decoder
 
vgg19 = models.vgg16(pretrained=True)
vgg19 = nn.Sequential(*list(vgg19.children())[:-1])
 
class Network(nn.Module):
    def create_emb_layer(self,weights_matrix):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = True
        return emb_layer
        
    def __init__(self):
        super().__init__()        
 
        self.num_clusters = 8
        self.hidden_size = 300
        self.vlad_output = 4096
        self.num_layers = 1
        self.emb_size = 100
        self.output_nodes = 300
        self.ck = nn.Parameter(torch.rand(self.num_clusters, 512))
        
        self.encoder = vgg19
        self.oned_conv_layer = nn.Conv2d(512,self.num_clusters,kernel_size=1,stride=1,padding=0)       
        self.softmax = nn.Softmax(dim=1)
 
        self.embeddings = self.create_emb_layer(torch.from_numpy(initial_embeddings))
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers, batch_first=True)
        
        self.vlad_linear_layer = nn.Sequential(
            nn.Linear(self.vlad_output, self.hidden_size)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, len(unique_words) + 2)
        )
       
          
    def forward(self, x,y,training_mode):
        x = self.encoder(x)
        
        N, C = x.shape[:2]
        soft_assign = self.oned_conv_layer(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.ck[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  
        vlad = vlad.view(vlad.size(0), -1) 
        
        x = self.vlad_linear_layer(vlad)
        
        if training_mode == True:
            ws = torch.unsqueeze(self.embeddings(torch.tensor(y).cuda()),0).cuda()
            x = torch.unsqueeze(x,0) 
            
            x_out,(y_out,s_out) = self.lstm(ws,(x,x))
            outs =  self.output_layer(x_out.squeeze(0))
            return outs
        
        else:
            out = []
            out_i = []
            first = True
            count = 0
            while True:
                if count >= 40: 
                    return out,out_i
                else:
                    count += 1
    
                if not first:  
                    o = self.output_layer(y_out)
                    o = torch.squeeze(o,0)
                    v,i = torch.max(o,1)
                    i = i.item()
                    if i == 6471:
                        break
                    elif i == 6470:
                        out.append("st")
                        out_i.append(-1)
                    else:
                        out.append(unique_words[i])
                        out_i.append(i)
                else:
                    i = 6470
                    first = False
                    y_out = torch.unsqueeze(x,0).cuda()
                    s_out = y_out

                if i != 6471:
                    if i == 6470:
                      _,(y_out,s_out) = self.lstm(torch.unsqueeze(torch.unsqueeze(self.embeddings(torch.tensor(i).cuda()),0),0),(y_out,s_out))
                    else:
                      _,(y_out,s_out) = self.lstm(torch.unsqueeze(torch.unsqueeze(self.embeddings(torch.tensor(i).cuda()),0),0),(y_out,s_out))
                else:
                    break
            return out,out_i

# Training 
 
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
print("using {}".format(dev))
 
model = Network()
model.to(torch.device(dev))
criterion = nn.CrossEntropyLoss().to(torch.device(dev))
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)

train_from_saved_model = True
start_epoch = 150
num_epochs = 200
 
if train_from_saved_model:
    checkpoint = torch.load("model/model_image_captioning_lstm_{}.pth".format(start_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
for e in range(start_epoch + 1,start_epoch + num_epochs + 1):
    epoch_loss = 0
    batch_size = 1
    d = 0
    prev = 0 
    print(e)
    for it in range(int(3200/batch_size)):
        loss = 0
        optimizer.zero_grad()
        for i in range(batch_size):
          optimizer.zero_grad()
          expected = re.split(" |\n",captions[(it * batch_size)+ i , 0])[:-1]
          expected = [len(unique_words)] + [unique_words.index(s.lower()) for s in expected]+ [len(unique_words) + 1]
          e_out = np.array(expected[1:])
          output = model(torch.squeeze(torch.from_numpy(features[it*batch_size + i:(it*batch_size) + (i+1),:,:,:,:]),1).float().cuda(),np.array(expected[:-1]),True)        
          output = output.squeeze(1).squeeze(1)
          et = torch.from_numpy(e_out).cuda()
          loss += criterion(torch.squeeze(output,1),et) 
          if it > 0 and it % 100 == 0:
            print(epoch_loss/it)
          epoch_loss += loss
          loss *= 10
          loss.backward()
          optimizer.step()
        d += batch_size       
    print("error:{}".format(epoch_loss/3200))
 
    if e % 3 == 0:
      torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': epoch_loss,
          }, "model/model_image_captioning_lstm_{}.pth".format(e))
      print("saved ",e)

# Testing 

smoothie = SmoothingFunction().method1
 
model = Network().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)
 
start_epoch = 207
checkpoint = torch.load("model/model_image_captioning_lstm_{}.pth".format(start_epoch))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
model.eval()

tot_score_1 = 0
tot_score_2 = 0
tot_score_3 = 0
tot_score_4 = 0 
refs = []
candidates = []
d = 0
div = 00
for f in features:
    print(d)
    if d >= div:
        expected = re.split(" |\n",captions[d,random.randint(0,4)])[:-1]
        expected = np.array([0,0])
        output,out_i = model(torch.from_numpy(f).float().cuda(),expected,False)        
        plt.imshow(np.moveaxis(np.squeeze(f,0),0,-1))
        plt.show()        
        
        exp = []
        for i in range(5):
          exp.append(re.split(" |\n",captions[d,i])[:-1])
        
        if len(out_i) >= 2:
          tot_score_1 += sentence_bleu(exp, output,weights=(1, 0, 0, 0),smoothing_function=smoothie)
          tot_score_2 += sentence_bleu(exp, output,weights=(0.5, 0.5, 0, 0),smoothing_function=smoothie)
          tot_score_3 += sentence_bleu(exp, output,weights=(0.33, 0.33, 0.33, 0),smoothing_function=smoothie)
          tot_score_4 += sentence_bleu(exp, output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoothie)
          div += 1
        
    d += 1

print(div)
print(tot_score_1/div)
print(tot_score_2/div)
print(tot_score_3/div)
print(tot_score_4/div)