import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.filters import threshold_otsu
from skimage.transform import resize
import os
from skimage.filters import sobel
from keras.models import load_model
from sys import argv

"""Here i used the best model(but not the most complex) with strongest photo processing"""

for i, file in enumerate(os.listdir('Photos')):
    """Folder Photos must in same directory"""
    model = load_model('model.h5')

    print(model.summary())

    ph = plt.imread(f'Photos\{file}')

    ph = color.rgb2gray(ph)

    ph = sobel(ph)

    ph = resize(ph, (1, 150, 150, 3))

    thres = threshold_otsu(ph)

    binary = ph > thres
    print(ph.shape)

    result = model.predict_on_batch(ph)

    proba = model.predict_proba(ph, verbose=False, batch_size=1)

    os.rename(f'Photos\{file}', f'Photos\horse_{i}.{file[-3:]}') if result[0] < 0.5 else os.rename(f'Photos\{file}',f'Photos\human{i}.{file[-3:]}')
