import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from skimage import color
from skimage.filters import threshold_otsu
from skimage.transform import resize
import os
from skimage.filters import sobel
from keras.models import load_model
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" On validation set highest score(89.8%) was achieved with rgb photos and 2 convolutional layers
    But when i started using real-life photos of them this network failed (predictions were consistent but with ~50%accuracy)
    Most upgraded model_heavy is probably good but more neurons needed(even this model with 5 epochs to fit was training for 30 minutess
    so continue upgrading iteratively is impossible(even test score was 87% while others have ~100%))"""

def processing(directory):
    photo_array = []

    for file in os.listdir(f'{directory}/horses'):
        photo = plt.imread(f'{directory}/horses/{file}')

        photo = color.rgb2gray(photo)

        photo = sobel(photo)

        photo = resize(photo, (150, 150, 1))

        photo_array.append(photo)

    photo_array = np.array(photo_array)

    X_horse = photo_array

    y_horse = np.zeros(len(X_horse))

    photo_array = []

    for file in os.listdir(f'{directory}/humans'):
        photo = plt.imread(f'{directory}/humans/{file}')

        photo = color.rgb2gray(photo)

        photo = sobel(photo)

        photo = resize(photo, (150, 150, 1))

        photo_array.append(photo)

    photo_array = np.array(photo_array)

    X_human = photo_array

    y_human = np.ones(len(X_human))

    X, y = np.concatenate((X_horse, X_human)), np.concatenate((y_horse, y_human))
    print(X_human.shape, X_horse.shape)
    print(X.shape, y.shape)
    print(y[0:5], y[-5:])
    return X, y

X, y = processing('train')



from keras import Sequential
from keras import layers
model = Sequential()

model.add(layers.Conv2D(5, kernel_size=(50,50), activation='relu', input_shape=(150, 150, 1)))

model.add(layers.Dropout(0.3))

model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dropout(0.3))

model.add(layers.BatchNormalization())

model.add(layers.Dense(50, activation='relu'))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(20, activation='relu'))

model.add(layers.Dense(20, activation='relu'))

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=5, batch_size=5, verbose=True)

model.save('model_main.h5')


X_val, y_val = processing('validation')

score = model.evaluate(X_val,y_val)

print(f'Score{score}')
