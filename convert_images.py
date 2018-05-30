# coding=utf-8

from __future__ import print_function

import pickle

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import pandas as pd
import tifffile as tiff
import random

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('dae_server', 'localhost:6007',
                           'predictionService host:port')

tf.app.flags.DEFINE_string('fc_server', 'localhost:6008',
                           'predictionService host:port')

tf.app.flags.DEFINE_integer('image_size', 32, '')
tf.app.flags.DEFINE_integer('channels', 4, '')
tf.app.flags.DEFINE_integer('mid_hidden_size', 4*4*64, '')


FLAGS = tf.app.flags.FLAGS

FILE_2015 = '/home/zyt/data/TianChi/preliminary/quickbird2015.tif'
FILE_2017 = '/home/zyt/data/TianChi/preliminary/quickbird2017.tif'


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w*h, d]).astype(np.float32)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 0, axis=0)
    maxs = np.percentile(matrix, 100, axis=0)
    lens = maxs - mins
    if np.min(lens) == 0:
        return np.reshape(matrix, [w, h, d])

    matrix = (matrix - mins[None, :]) / lens[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def dae_server(raw_image):
    host, port = FLAGS.dae_server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'dae'
    request.model_spec.signature_name = 'predict_signature'

    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(raw_image,
                                                                       shape=[1, FLAGS.image_size, FLAGS.image_size, FLAGS.channels]))
    result = stub.Predict(request, 10.0)
    return result.outputs['mid_hidden'].float_val


def fc_server(x):
    host, port = FLAGS.fc_server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'fc'
    request.model_spec.signature_name = 'predict_signature'
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(x,
                                                                       shape=[1, 4*4*64*2]))   # the size of input:[batch_size, mid_hidden_size*2]
    result = stub.Predict(request, 10.0)
    return result.outputs['output'].float_val


def convert_image(im_2015, im_2017):
    w, h, d = im_2015.shape
    res_image = np.zeros(shape=[w, h], dtype=np.uint8)
    fc_input = np.zeros(shape=[2*FLAGS.mid_hidden_size], dtype=np.float32)
    # the size of processed image is 32
    xLen = 0
    yLen = 0
    for xstart, xend in zip(range(0, w, 16), range(32, w, 16)):
        xLen += 1
    for ystart, yend in zip(range(0, h, 16), range(32, h, 16)):
        yLen += 1
    isChange = np.zeros(shape=[xLen, yLen])

    xLen = 0
    for xstart, xend in zip(range(0, w, 16), range(32, w, 16)):
        for ystart, yend in zip(range(0, h, 16), range(32, h, 16)):
            yLen = 0
            image_2015 = np.array(scale_percentile(im_2015[xstart:xend, ystart:yend, :]), dtype=np.float32)
            image_2017 = np.array(scale_percentile(im_2017[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2015 = dae_server(image_2015)
            hidden_2017 = dae_server(image_2017)
            fc_input[0:FLAGS.mid_hidden_size] = np.reshape(hidden_2015, [FLAGS.mid_hidden_size])
            fc_input[FLAGS.mid_hidden_size:] = np.reshape(hidden_2017, [FLAGS.mid_hidden_size])
            fc_output = np.array(fc_server(fc_input))
            print("x at", xstart, "y at", ystart, "fc_output is", fc_output)
            if (fc_output - 0.9) > 0:
                print("x at", xstart, "y at", ystart, ", convert image.")
                res_image[xstart:xend, ystart:yend] = 1
                isChange[xLen, yLen] = 1
            else:
                if isChange[xLen, yLen-1] == 1 and isChange[xLen-1, yLen-2] == 1:
                    res_image[xstart:xend, ystart:yend] = 0
                if isChange[xLen-1, yLen] == 1 and (isChange[xLen-12, yLen-1] == 1 or isChange[xLen-2, yLen+1] == 1):
                    res_image[xstart:xend, ystart:yend] = 0

            yLen += 1
        xLen += 1

    # delete single 32*32 point
    for xstart, xend in zip(range(0+16, w-16, 16), range(32+16, w-16, 16)):
        for ystart, yend in zip(range(0+16, h-16, 16), range(32+16, h-16, 16)):
            if np.sum(res_image[xstart:xend, ystart:yend]) > 0:
                if np.sum(res_image[xstart-16:xend+16, ystart-16:yend+16]) - np.sum(res_image[xstart:xend, ystart:yend]) <= 0.01:
                    res_image[xstart:xend, ystart:yend] = 0

    print("convert image successfully!")
    output_file = open("/home/zyt/data/TianChi/result/result1008.tif", "wb")
    tiff.imsave(output_file, res_image)
    output_file.close()
    print("save result successfully!")


def main(_):
    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)
    convert_image(im_2015, im_2017)

    left_2015 = im_2015[0:600, 0:9000, :]
    left_2017 = im_2017[0:600, 0:9000, :]
    #convert_image(left_2015, left_2017)

    tiny_2015 = im_2015[600:2100, 9600:11300, :]
    tiny_2017 = im_2017[600:2100, 9600:11300, :]
    #convert_image(tiny_2015, tiny_2017)

    tiny_2015_1 = im_2015[2800:3500, 5600:6000, :]
    tiny_2017_1 = im_2017[2800:3500, 5600:6000, :]
    #convert_image(tiny_2015_1, tiny_2017_1)

    tiny_2015_2 = im_2015[600:1800, 10200:11200, :]
    tiny_2017_2 = im_2017[600:1800, 10200:11200, :]
    #convert_image(tiny_2015_2, tiny_2017_2)

    tiny_2015_3 = im_2015[4100:4700, 11400:12000, :]
    tiny_2017_3 = im_2017[4100:4700, 11400:12000, :]
    #convert_image(tiny_2015_3, tiny_2017_3)

    tiny_2015_4 = im_2015[4000:4800, 900:2100, :]
    tiny_2017_4 = im_2017[4000:4800, 900:2100, :]
    #convert_image(tiny_2015_4, tiny_2017_4)

    tiny_2015_5 = im_2015[400:800, 2500:3000, :]
    tiny_2017_5 = im_2017[400:800, 2500:3000, :]
    #convert_image(tiny_2015_5, tiny_2017_5)

if __name__ == '__main__':
    tf.app.run()
