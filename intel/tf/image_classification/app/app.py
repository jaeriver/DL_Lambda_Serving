import sys
import os
import json
import string
import time
import io
import requests
import numpy as np

# Importing TensorFlow
import tensorflow as tf
import tensorflow_hub as hub

# Loading model
# model_path = './model/'
# loaded_model = tf.saved_model.load(model_path)
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
  hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

def make_dataset(batch_size, size):
    image_shape = (size, size, 3)

    data = np.random.uniform(-1, 1, size=image_shape).astype("float32")

    return data,image_shape

def handler(event, context):
    batch_size = event['batch_size']
    size = event['size']
    data,image_shape = make_dataset(batch_size, size)

    # Executing inference.
    converted_img  = tf.image.convert_image_dtype(data, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = classifier.predict(converted_img)
    end_time = time.time()

    obj = {
        "result":result
    }    
    print(obj)
    return True
