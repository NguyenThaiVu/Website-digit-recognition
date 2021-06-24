from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *

import numpy as np
import cv2
from sklearn.utils import shuffle
from imutils import build_montages
import h5py
import os
import scipy.io as sio
import urllib.request
import shutil

from tensorflow.keras.datasets import mnist
import copy
import cv2
from PIL import Image

from tensorflow import keras 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import  Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers

from .help_function import *


# ===================================DIGIT RECOGNITION MODEL =================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

MODEL_PATH = "digit_recognize.h5"
model = load_model(MODEL_PATH)

model.summary()

# Standardize preprocessor
normPro = NomalizePreprocessor()

# Create your views here.
def image_view(request):

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, 'image_form.html', {'form' : form})


def success(request):

    # List face in this image
    face_image = []

    # Home button
    if request.method == 'POST':
        return redirect('image_view')

    # Extract image
    images = MyImage.objects.all()

    image = images[len(images) - 1]

    image_numpyArray = np.array(Image.open(image.img))

    originalImage = copy.deepcopy(image_numpyArray)

    # Predict image
    image_numpyArray = cv2.cvtColor(image_numpyArray, cv2.COLOR_BGR2GRAY)
    image_numpyArray = cv2.resize(image_numpyArray, (32, 32))

    image_numpyArray = np.expand_dims(image_numpyArray, 0)
    image_numpyArray = np.expand_dims(image_numpyArray, -1)

    image_numpyArray = normPro.preprocess(image_numpyArray)

    predict = model.predict(image_numpyArray)

    label_predict = np.argmax(predict)
    confidence = np.max(predict)

    pil_image = to_image(originalImage)
    resultImage = to_data_uri(pil_image)

    return render(request, 'display_image.html', {'resultImage' : resultImage, 'label': label_predict})