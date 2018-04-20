from __future__ import print_function

import os

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# This is a placeholder for a Google-internal import.


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _transform_feature(value):
    if is_number(value):
        return _float_feature(float(value))
    else:
        return _bytes_feature(value.encode())


def do_inference(query_map={}):
    wide_and_deep_serving_url = os.environ.get('WIDE_AND_DEEP_SERVING_URL', '10.206.8.93:19000')
    host, port = wide_and_deep_serving_url.split(':')

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'movielens'
    request.model_spec.signature_name = 'serving_default'

    numeric_query_map = {
        'user': 0,
        'item': 0,
        'age': 30,
        'delay_days': 430,
        'watch_year': 1998,
        'watch_month': 3,
        'watch_day': 7,
        'watch_wd': 6,
        'watch_season': 0,
        'relase_year': 1997,
        'release_month': 1,
        'release_day': 1,
        'release_wd': 1,
        'age_span': 30
    }

    str_feature_map = {
        'watch_span': 'TenYear',
        'gender': 'F',
        'occupation': 'doctor',
        'zipcode': '60089',
        'action': '0',
        'adventure': '0',
        'animation': '0',
        'child': '0',
        'comedy': '0',
        'crime': '0',
        'documentary': '0',
        'drama': '0',
        'fantasy': '0',
        'film_noir': '0',
        'horror': '0',
        'musical': '0',
        'mystery': '0',
        'romance': '0',
        'sci_fi': '0',
        'thriller': '0',
        'war': '0',
        'western': '0'
    }

    feature_dict = {}
    for key in numeric_query_map.keys():
        feature_dict[key] = _float_feature(float(query_map.get(key, numeric_query_map.get(key))))
    for key in str_feature_map.keys():
        feature_dict[key] = _bytes_feature(query_map.get(key, str_feature_map.get(key)).encode())

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(serialized, shape=[1]))
    result_feature = stub.Predict.future(request, 10.0)
    prediction = result_feature.result().outputs['outputs'].float_val[0]

    return {
        'score': prediction,
        'request': query_map
    }
