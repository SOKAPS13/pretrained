#!/usr/bin/env python
# coding: utf-8

# In[17]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define the image size and paths to the train, validation, and test directories
IMAGE_SIZE = [224, 224]
train_path = r'C:\Users\Abhishek\Downloads\archive (23)\Cotton Disease\train'
valid_path =r'C:\Users\Abhishek\Downloads\archive (23)\Cotton Disease\val'
test_path = r'C:\Users\Abhishek\Downloads\archive (23)\Cotton Disease\test'

# Use the InceptionV3 pre-trained model with the weights from ImageNet
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all the layers in the pre-trained model
for layer in inception.layers:
    layer.trainable = False

# Get the number of output classes
folders = glob(train_path + '/*')

# Flatten the output from the pre-trained model
x = Flatten()(inception.output)

# Add a fully connected layer with a softmax activation function for multiclass classification
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object using the input and output layers
model = Model(inputs=inception.input, outputs=prediction)

# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use ImageDataGenerator to import images and perform data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
valid_set = valid_datagen.flow_from_directory(valid_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train the model using the train and validation datasets
r = model.fit_generator(
  train_set,
  validation_data=valid_set,
  epochs=20,
  steps_per_epoch=len(train_set),
  validation_steps=len(valid_set)
)

# Plot the training and validation loss and accuracy
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

# Evaluate the model using the test dataset
test_loss, test_acc = model.evaluate_generator(test_set)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# In[ ]:




