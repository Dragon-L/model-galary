from __future__ import print_function

import sys

import io
import requests
from scipy import misc

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def get_image(url):
    response = requests.get(url)
    image = misc.imread(io.BytesIO(response.content))
    jpeg_bytes = io.BytesIO()
    misc.imsave(jpeg_bytes, image, format='JPEG')
    return jpeg_bytes.getvalue()


def do_inference(url):
    host, port = 'ai02:9001'.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict_images'

    image = get_image(url)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1]))

    result = stub.Predict(request, 10.0)  # 10 seconds
    result = dict(zip(
        map(lambda byteStr: byteStr.decode(), list(result.outputs['classes'].string_val)),
        list(result.outputs['scores'].float_val)
    ))
    return result
