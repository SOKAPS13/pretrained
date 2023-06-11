#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Directories where your data is stored
train_dir = r'C:\Users\Abhishek\Desktop\New folder (2)\Paddy\train'
validation_dir = r'C:\Users\Abhishek\Desktop\New folder (2)\Paddy\valid'
test_dir = r'C:\Users\Abhishek\Desktop\New folder (2)\Paddy\test'

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 8  # Reduced batch size
NUM_CLASSES = 3  # Number of classes in your dataset
EPOCHS = 100

# Generate batches of tensor image data with real-time data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Load base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add a new top layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Reduced the number of neurons
x = Dropout(0.2)(x)  # Add dropout layer to reduce overfitting
x = BatchNormalization()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# This is the model we will train
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr])

# Unfreeze the layers of the base model and fine-tune the entire model
for layer in base_model.layers:
    layer.trainable = True

# Recompile the model
model.compile(optimizer=Adam(lr=0.00001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Continue training the model
history_fine_tuning = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test data after fine-tuning
# Evaluate the model on the test data after fine-tuning
score = model.evaluate(test_generator)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# In[ ]:




