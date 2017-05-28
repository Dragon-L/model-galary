from __future__ import print_function

import io
import requests
from scipy import misc
import base64

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def get_image(url):
    img = requests.get(url)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(img.content).decode()
    img = misc.imread(io.BytesIO(img.content), True)
    img = (misc.imresize(img, size=[28, 28]) / 255.0).astype(numpy.float32)
    return img, img_b64


def do_inference(url):
    host, port = 'ai02:9000'.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'

    image, image_b64 = get_image(url)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, image.size]))

    result = stub.Predict(request, 5.0)  # 5 seconds

    response = numpy.array(result.outputs['scores'].float_val)
    prediction = numpy.argmax(response)

    return {
        'label': int(prediction),
        'image_b64': image_b64
    }
