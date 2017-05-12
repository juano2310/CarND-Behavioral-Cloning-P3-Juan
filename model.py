import os
import csv

samples = []
with open('./data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import sklearn
sklearn.utils.shuffle(samples)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Number of traing samples: ",len(train_samples))
print("Number of validation samples: ",len(validation_samples))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
            	path = os.path.normpath(batch_sample[1]).split(os.path.sep) #name = './data2/IMG/'+path[0].split('\\')[-1]
            	name = './data3/IMG/'+path[0].split('\\')[-1] # name = './data2/IMG/'+batch_sample[0].split('/')[-1]
            	left_image = cv2.imread(name)
            	left_angle = float(batch_sample[3]) + 0.15
            	images.append(left_image)
            	angles.append(left_angle)
            	flip_left_image = np.fliplr(left_image)
            	flip_left_angle = -1 * left_angle
            	images.append(flip_left_image)
            	angles.append(flip_left_angle)
            	path = os.path.normpath(batch_sample[2]).split(os.path.sep) #name = './data2/IMG/'+path[0].split('\\')[-1]
            	name = './data3/IMG/'+path[0].split('\\')[-1] # name = './data2/IMG/'+batch_sample[0].split('/')[-1]
            	right_image = cv2.imread(name)
            	right_angle = float(batch_sample[3]) - 0.15
            	images.append(right_image)
            	angles.append(right_angle)
            	flip_right_image = np.fliplr(right_image)
            	flip_right_angle = -1 * right_angle
            	images.append(flip_right_image)
            	angles.append(flip_right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size_value = 32
n_epoch = 15

train_generator = generator(train_samples, batch_size=batch_size_value)
validation_generator = generator(validation_samples, batch_size=batch_size_value)

model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((50,20), (0,10)), input_shape=(160,320,3)))

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Nvidia model
model.add(Conv2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))
model.add(SpatialDropout2D(.5, dim_ordering='default'))

model.add(Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# checkpoint
filepath="weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

# Fit the model
history_object = model.fit_generator(train_generator, steps_per_epoch=(len(train_samples) / batch_size_value), validation_data=validation_generator, validation_steps=(len(validation_samples)/batch_size_value), callbacks=callbacks_list, epochs=n_epoch)

# Save model
model.save('model.h5')
with open('model.json', 'w') as output_json:
    output_json.write(model.to_json())

# Save TensorFlow model
tf.train.write_graph(K.get_session().graph.as_graph_def(), logdir='.', name='model.pb', as_text=False)

# Plot the training and validation loss for each epoch
print('Generating loss chart...')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model.png')

# Done
print('Done.')
