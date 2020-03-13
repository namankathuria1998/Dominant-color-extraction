#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[142]:


model=ResNet50(weights='imagenet',input_shape=(224,224,3))
model_new=Model(model.input,model.layers[-2].output)
model_new._make_predict_function()

# In[143]:


def preprocess_image(img_path):
    img=image.load_img(img_path,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
#     print(img.shape)
    return img


# In[144]:


def encode_image(img_path):
    img=preprocess_image(img_path)
    feature_vector=model_new.predict(img)
    feature_vector=feature_vector.reshape((1,2048))
    return feature_vector


# In[145]:


with open("./word_to_idx.pkl","rb") as f:
    word_to_index = pickle.load(f)

with open("./index_to_word.pkl","rb") as f:
    index_to_word = pickle.load(f)


# In[146]:


model = load_model("../../../myimagecaptioningmodel.h5")
model._make_predict_function()

# In[147]:


model


# In[148]:


def predict_caption(photo):
    in_text="startseq "
    max_len=35
    for i in range(max_len):
        sequence=[word_to_index[word] for word in in_text.split() if word in word_to_index]
        sequence=pad_sequences([sequence],maxlen=max_len,padding="post")
        
        probdist=model.predict([photo,sequence])
        index=np.argmax(probdist)
        wordpredicted=index_to_word[index]
        
        in_text += (wordpredicted + " ")
        
        if wordpredicted == "endseq":
            break
     
    in_text=in_text.split()
    in_text=in_text[1:-1]
    in_text=" ".join(in_text)
    return in_text


# In[155]:


# encoding = encode_image("./dogrunning.jpg")
# caption=predict_nextword(encoding)
# print(caption)


# In[156]:


# encoding


# In[157]:


# encoding.shape


# In[158]:


# caption = predict_caption(encoding)
# print(caption)


# In[159]:


len(word_to_index)


# In[160]:


def caption_this_image(image):
    encoding = encode_image(image)
    caption = predict_caption(encoding)
    return caption


# In[ ]:




