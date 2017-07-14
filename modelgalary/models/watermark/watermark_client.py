import base64
import io
import os

import numpy
import requests
import tensorflow as tf
from grpc.beta import implementations
from scipy import misc

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


def get_image(url):
    img = requests.get(url)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(img.content).decode()
    image = misc.imread(io.BytesIO(img.content))
    jpeg_bytes = io.BytesIO()
    misc.imsave(jpeg_bytes, image, format='JPEG')
    return jpeg_bytes.getvalue(), img_b64


def do_inference(url):
    mnist_serving_url = os.environ.get('WATERMARK_SERVING_URL', 'ai02:9002')
    host, port = mnist_serving_url.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'watermark'
    request.model_spec.signature_name = 'classified'

    image, image_b64 = get_image(url)

    feature_dict = {
        'images/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['input_example_tensor'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

    result = stub.Predict(request, 10.0)  # 10 seconds
    result = zip(
        map(lambda byte_str: byte_str.decode(), list(result.outputs['classes:0'].string_val)),
        map(float, list(result.outputs['scores:0'].float_val))
    )

    def to_dict(name_prob_tuple):
        print(list(name_prob_tuple))
        return {'name': name_prob_tuple[0], 'probability': name_prob_tuple[1]}

    return {
        'image_b64': image_b64,
        'labels': list(map(to_dict, result))
    }

if __name__ == '__main__':
    do_inference("http://localhost:3001/assets/watermark/text_1.jpg")
