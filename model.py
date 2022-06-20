# Use these code below if you have wandb account
# !wandb login b9f226664a5b9170f18e1e0f01bae6a2b06a58d5

# import wandb
# from wandb.keras import WandbCallback
# wandb.init(project="emotion-detection", entity="pen215")

# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 64
# }


import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import glob, os

from tensorflow.keras.utils import to_categorical


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

data = pd.read_csv("data\\fer2013.csv")

train_set = data[data.Usage=='Training']
val_set = data[data.Usage=='PublicTest']
test_set = data[data.Usage=='PrivateTest']


xtrain = np.array(list(map(str.split, train_set.pixels)), np.float32)
xval = np.array(list(map(str.split, val_set.pixels)), np.float32)
xtest = np.array(list(map(str.split, test_set.pixels)), np.float32)

xtrain = xtrain.reshape(xtrain.shape[0], 48,48,1) / 255
xval = xval.reshape(xval.shape[0], 48,48,1) / 255
xtest = xtest.reshape(xtest.shape[0], 48,48,1) / 255


classes = 7
ytrain = train_set.emotion
ytrain = to_categorical(ytrain, classes)

yval = val_set.emotion
yval = to_categorical(yval, classes)

ytest = test_set.emotion
ytest = to_categorical(ytest, classes)


# history = model.fit(xtrain, ytrain, validation_data=(xval, yval),epochs=10, batch_size=64, callbacks=[WandbCallback()])
history = model.fit(xtrain, ytrain, validation_data=(xval, yval),epochs=10, batch_size=64)




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')


test_loss, test_acc = model.evaluate(xtest, ytest)
print('test_acc:', test_acc)
print('test_loss:', test_loss)
