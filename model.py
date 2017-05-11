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
            	path = os.path.normpath(batch_sample[2]).split(os.path.sep) #name = './data2/IMG/'+path[0].split('\\')[-1]
            	name = './data3/IMG/'+path[0].split('\\')[-1] # name = './data2/IMG/'+batch_sample[0].split('/')[-1]
            	right_image = cv2.imread(name)
            	right_angle = float(batch_sample[3]) - 0.15
            	images.append(right_image)
            	angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam

model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((52,25), (10,10)), input_shape=(160,320,3)))

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Nvidia model
model.add(Conv2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=(len(train_samples) / 32), validation_data=validation_generator, validation_steps=(len(validation_samples)/32), epochs=10) #, verbose =2)
model.save('model.h5')
