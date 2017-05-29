from __future__ import print_function

import os
import io
import requests
import base64
from scipy import misc

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def get_image(url):
    img = requests.get(url)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(img.content).decode()
    image = misc.imread(io.BytesIO(img.content))
    jpeg_bytes = io.BytesIO()
    misc.imsave(jpeg_bytes, image, format='JPEG')
    return jpeg_bytes.getvalue(), img_b64


def do_inference(url):
    mnist_serving_url = os.environ.get('INCEPTION_SERVING_URL', 'ai02:9001')
    host, port = mnist_serving_url.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict_images'

    image, image_b64 = get_image(url)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1]))

    result = stub.Predict(request, 10.0)  # 10 seconds
    result = zip(
        map(lambda byte_str: byte_str.decode(), list(result.outputs['classes'].string_val)),
        map(float, list(result.outputs['scores'].float_val))
    )
    to_dict = lambda name_prob_tuple: {'name': name_prob_tuple[0], 'probability': name_prob_tuple[1]}
    return {
        'image_b64': image_b64,
        'labels': list(map(to_dict, result))
    }
