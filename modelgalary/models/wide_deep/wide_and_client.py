from __future__ import print_function

import os

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def do_inference(query_map={}):
    wide_and_deep_serving_url = os.environ.get('WIDE_AND_DEEP_SERVING_URL', 'ai03:28888')
    host, port = wide_and_deep_serving_url.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'wide_and_deep'
    request.model_spec.signature_name = 'serving_default'

    feature_dict = {'age': _float_feature(value=query_map.get('age', 25)),
                    'capital_gain': _float_feature(value=query_map.get('capital_gain', 0)),
                    'capital_loss': _float_feature(value=query_map.get('capital_loss', 0)),
                    'education': _bytes_feature(value=query_map.get('education', '11th').encode()),
                    'education_num': _float_feature(value=query_map.get('education_num', 7)),
                    'gender': _bytes_feature(value=query_map.get('gender', 'Male').encode()),
                    'hours_per_week': _float_feature(value=query_map.get('hours_per_week', 40)),
                    'native_country': _bytes_feature(value=query_map.get('native_country', 'United-States').encode()),
                    'occupation': _bytes_feature(value=query_map.get('occupation', 'Machine-op-inspct').encode()),
                    'relationship': _bytes_feature(value=query_map.get('relationship', 'Own-child').encode()),
                    'workclass': _bytes_feature(value=query_map.get('workclass', 'Private').encode())}

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

    result_feature = stub.Predict.future(request, 10.0)
    prediction = result_feature.result().outputs['scores']

    return {'max_scores': np.argmax(prediction.float_val)}
