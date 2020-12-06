# Preprocessing data

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

# Extracting images and corresponding references [5 references per image]
  
f = open("captions.txt","r")
f = f.readlines()
image_names = open("image_names.txt","r")
image_names = image_names.readlines()
new = []
for image in image_names:
  new.append(image.split("\n")[0])
image_names = new  
images = []
references = []
reference = []
i = 0
flag = False
count = 0
image_not = 0
for line in f:
  count += 1
  print(count)
  if i == 5: 
    i = 0
    reference = []    
  
  temp = line.split("#")
  if i == 0:
    if temp[0] in image_names: 
      images.append(temp[0])
      flag = True
    else:
      image_not += 1
      flag = False
  reference.append(temp[1].split("\t",1)[1])
  
  if i == 4:
    if flag == True:
      references.append(np.array(reference))
    
  i += 1
references = np.array(references)
images = np.array(images)
np.save("references.npy",references)
np.save("images.npy",images)
print(references.shape,images.shape)
print(image_not)

# Extracting unique words : vocabulary

captions = np.load("references.npy")
words = []
for captions_image in captions:
  for caption in captions_image:
    for t in re.split(" |\n",caption)[:-1]:
      words.append(t.lower())
words = np.array(words).flatten()
words = np.unique(words)
np.save("words.npy",words)

# Formatting glove embeddings
 
words = []
idx = 0
word2idx = {}
vectors = []
 
with open('../glove.6B.300d.txt', 'r') as f:
    for l in f:
        line = l.split(" ")
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
 
pickle.dump(vectors, open('../6B.300_embeddings.pkl', 'wb'))    
pickle.dump(words, open('../6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('../6B.300_idx.pkl', 'wb'))

# Pre-processing for Image data
 
transform = transforms.Compose([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor()
])
 
images = np.load("images.npy")
features = []
count = 0
for image in images:
  print(count)
  count += 1
  img = Image.open("Images/" + image)
  print(img.shape)
  input()
  img_t = transform(img)
  batch_t = torch.unsqueeze(img_t, 0)
  features.append(batch_t.numpy())
features = np.array(features)
# np.save("features2.npy",features)
# print(features.shape)

# Creating initial embeddings for all words in the vocabulary 
 
input_file = open('../6B.300_embeddings.pkl', "rb")
glove = pickle.load(input_file)
idx_file = open('../6B.300_idx.pkl', "rb")
idx = pickle.load(idx_file)
target_vocab = np.load("words.npy")
matrix_len = target_vocab.shape[0]
weights_matrix = np.zeros((matrix_len + 2, 300))
words_found = 0
for i in range(matrix_len):
    print(i)
    try:
        weights_matrix[i] = glove[idx[target_vocab[i]]]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
weights_matrix[matrix_len] = np.random.normal(scale=0.6, size=(300, ))
weights_matrix[matrix_len+1] = np.random.normal(scale=0.6, size=(300, ))
np.save("initial_embeddings.npy",weights_matrix)



inp = np.load("initial_embeddings.npy")
from sklearn import preprocessing
out = preprocessing.normalize(inp)
np.save("initial_embeddings.npy",out)