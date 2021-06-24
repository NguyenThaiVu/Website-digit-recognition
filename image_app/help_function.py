import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


# Convert from numpy array to URI image 
def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img

import base64
from io import BytesIO
def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 


# Nomalize preprocessor 
class NomalizePreprocessor:
    def __init__(self):
        pass

    def preprocess(self, image):
        return image.astype(np.float32) / 255.0
