#!/usr/bin/env python
# coding: utf-8

# In[1]:



from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
import os
import pandas as pd
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import natsort
from PIL import Image
from tqdm import tqdm
from re import search


# In[27]:


DIR=r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\images'


# In[28]:


train=pd.read_csv(r"C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train.csv")
test=pd.read_csv(r"C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\test.csv")


# In[29]:


train.head()


# In[30]:


test.head()


# In[31]:


image1=Image.open(r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\images\Train_922.jpg')
plt.imshow(image1)
plt.show()


# In[32]:


#preparing the model


# In[33]:


class_names=train.loc[:,'healthy':].columns
print(class_names)


# In[34]:


number=0
train['label']=0
for i in class_names:
    train['label']=train['label'] + train[i] * number
    number=number+1
    #if hat type of disease available show 1 
    # otherwise 0
    #healthy=0;multiple disease=1,rust=2,scab=3


# In[35]:


train.head()


# In[36]:


DIR


# In[37]:


natsort.natsorted(os.listdir(DIR))#sorting the images


# In[38]:


def get_label_img(img):
    if search("Train",img):
        img=img.split('.')[0]
        label=train.loc[train['image_id']==img]['label']
        return label


# In[39]:


def create_train_data():
    images=natsort.natsorted(os.listdir(DIR))
    for img in tqdm(images):
        label=get_label_img(img)
        path=os.path.join(DIR,img)
        
        if search("Train",img):
            if (img.split("_")[1].split(".")[0]) and label.item()==0:
                shutil.copy(path,r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train\healthy')
            elif(img.split("_")[1].split(".")[0]) and label.item()==1:
                shutil.copy(path,r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train\multiple_disease')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==2:
                shutil.copy(path,r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train\rust')
                
            elif(img.split("_")[1].split(".")[0]) and label.item()==3:
                shutil.copy(path,r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train\scab')
                
        elif search("Test",img):
            shutil.copy(path,r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\test')
                


# In[40]:


train_dir=create_train_data()


# In[41]:


Train_DIR=r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train'
Categories=['healthy','multiple_disease','rust','scab']
for j in Categories:
    path=os.path.join(Train_DIR,j)
    for img in os.listdir(path):
        old_image=cv2.imread(os.path.join(path,img),cv2.COLOR_BGR2RGB)
        plt.imshow(old_image)
        plt.show()
        break
    break


# In[42]:


IMG_SIZE=224
new_image=cv2.resize(old_image,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_image)
plt.show()


# In[18]:


train_labels = np.float32(train.loc[:, 'healthy':'scab'].values)


# In[19]:


train_labels


# In[20]:


train, val = train_test_split(train, test_size = 0.15)


# In[21]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( horizontal_flip=True,
 vertical_flip=True,
 rotation_range=10,
 width_shift_range=0.1,
 height_shift_range=0.1,
 zoom_range=.1,
 fill_mode='nearest',
 shear_range=0.1,
 rescale=1/255,
 brightness_range=[0.5, 1.5])


# In[22]:


datagen=ImageDataGenerator(rescale=1./255,
                          shear_range=0.2,
                           zoom_range=0.2,
                     horizontal_flip=True,
                       vertical_flip=True,
                      validation_split=0.2)
train_datagen=datagen.flow_from_directory(r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train',
target_size=(IMG_SIZE,IMG_SIZE),
                  batch_size=16,
      class_mode='categorical',
              subset='training')
val_datagen=datagen.flow_from_directory(r'C:\Users\Abhishek\Downloads\plant-pathology-2020-fgvc7 (2)\train',
 target_size=(IMG_SIZE,IMG_SIZE),
                   batch_size=16,
        class_mode='categorical',
              subset='validation')


# In[23]:


from tensorflow.keras.applications.inception_v3 import InceptionV3

from keras.models import Model
import keras
from keras import optimizers
import tensorflow as tf


# In[24]:


pretrained_model =InceptionV3 (include_top=False, weights='imagenet', input_shape=(128,128,3))


# In[43]:


model = tf.keras.Sequential([
 pretrained_model,
 tf.keras.layers.GlobalAveragePooling2D(),
 tf.keras.layers.Dropout(0.3),


 tf.keras.layers.Dense(4, activation='softmax')
    ])


# In[44]:


model.compile(
 optimizer=tf.keras.optimizers.Adam(),
 loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)


# In[45]:


model.summary()


# In[46]:


from keras.callbacks import ReduceLROnPlateau


# In[47]:


history_1 = model.fit(train_datagen,steps_per_epoch=20,
                                epochs=50, validation_data=val_datagen,
                                validation_steps=100,
                                verbose = 1, callbacks=[ReduceLROnPlateau(monitor= 'loss', factor=0.3, patience=3, min_lr=0.000001)],
                                use_multiprocessing=False,
                                shuffle=True
                                )


# 

# In[ ]:




