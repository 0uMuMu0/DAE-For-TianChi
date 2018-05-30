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


tf.app.flags.DEFINE_string('server', 'localhost:6007',
                           'predictionService host:port')

tf.app.flags.DEFINE_integer('image_size', 32, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('mid_hidden_size', 4*4*64, '')
tf.app.flags.DEFINE_integer('channels', 4, '')

FLAGS = tf.app.flags.FLAGS


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
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'dae'
    request.model_spec.signature_name = 'predict_signature'

    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(raw_image,
                                                                       shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.channels]))
    result = stub.Predict(request, 10.0)
    return result.outputs['mid_hidden'].float_val


def dae_server_from_saver(raw_image):
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():

        ckpt = tf.train.get_checkpoint_state("/home/zyt/data/TianChi/DAE/log/log15_17_32/")
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            mid_hidden = sess.run(tf.get_default_graph().get_tensor_by_name('max_pooling2d_3/MaxPool:0'), feed_dict={'inputs:0': raw_image})
        return mid_hidden



def build_dataset():
    FILE_2015 = '/home/zyt/data/TianChi/preliminary/quickbird2015.tif'
    FILE_2017 = '/home/zyt/data/TianChi/preliminary/quickbird2017.tif'
    FILE_label = '/home/zyt/data/TianChi/label/label.tif'
    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)
    building = np.array(tiff.imread(FILE_label), dtype=np.float32)

    image_size = FLAGS.image_size
    channels = FLAGS.channels
    mid_hidden_size = FLAGS.mid_hidden_size
    # 2200:4600, 400:2800
    tiny_2015 = im_2015[2200:4600, 400:2800, :]
    tiny_2017 = im_2017[2200:4600, 400:2800, :]
    tiny_building = building[2200:4600, 400:2800]

    # 2800:3500, 5600:6000
    tiny_2015_1 = im_2015[2800:3500, 5600:6000, :]
    tiny_2017_1 = im_2017[2800:3500, 5600:6000, :]
    tiny_building_1 = building[2800:3500, 5600:6000]

    # 600:1800, 10200:11200
    tiny_2015_2 = im_2015[600:1800, 10200:11200, :]
    tiny_2017_2 = im_2017[600:1800, 10200:11200, :]
    tiny_building_2 = building[600:1800, 10200:11200]

    # 4000:4800, 900:2100
    tiny_2015_3 = im_2015[4000:4800, 900:2100, :]
    tiny_2017_3 = im_2017[4000:4800, 900:2100, :]
    tiny_building_3 = building[4000:4800, 900:2100]

    # 400:800, 2500:3000
    tiny_2015_4 = im_2015[400:800, 2500:3000, :]
    tiny_2017_4 = im_2017[400:800, 2500:3000, :]
    tiny_building_4 = building[400:800, 2500:3000]

    dataset_len = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            dataset_len += 1

    train_x = np.zeros([dataset_len, mid_hidden_size*2])
    train_y = np.zeros([dataset_len, 1])  # 1--if has building, 0-- if no building
    print("start building......")
    ids = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            image_2015 = np.array(scale_percentile(tiny_2015[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2015 = dae_server(image_2015)
            image_2017 = np.array(scale_percentile(tiny_2017[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2017 = dae_server(image_2017)
            train_x[ids, 0:mid_hidden_size] = np.reshape(hidden_2015, [mid_hidden_size])
            train_x[ids, mid_hidden_size:] = np.reshape(hidden_2017, [mid_hidden_size])
            labels = tiny_building[xstart:xend, ystart:yend]
            if np.sum(labels) >= (32*32/2):
                train_y[ids] = 1
            else:
                train_y[ids] = 0
            ids += 1

    dataset = {"train_x": train_x, "train_y": train_y}

    output_file = open("/home/zyt/data/TianChi/dataset/20171026/train_data_3.pkl", "wb")
    pickle.dump(dataset, output_file)
    output_file.close()

    print("build dataset successfully!")


def build_dataset_2():
    FILE_2015 = '/home/zyt/data/TianChi/preliminary/quickbird2015.tif'
    FILE_2017 = '/home/zyt/data/TianChi/preliminary/quickbird2017.tif'
    FILE_pos_label = '/home/zyt/data/TianChi/label/pos_label.tif'
    FILE_label = '/home/zyt/data/TianChi/label/label.tif'
    FILE_my_label = '/home/zyt/data/TianChi/label/label1023.tif'

    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)
    label = np.array(tiff.imread(FILE_label), dtype=np.float32)
    my_label = np.array(tiff.imread(FILE_my_label), dtype=np.float32)
    pos_label = np.array(tiff.imread(FILE_pos_label), dtype=np.float32)

    image_size = FLAGS.image_size
    channels = FLAGS.channels
    # define in the DAE15_17_32.py
    mid_hidden_size = FLAGS.mid_hidden_size

    # 0:600, 0:9000
    tiny_2015 = im_2015[0:600, 0:9000, :]
    tiny_2017 = im_2017[0:600, 0:9000, :]
    tiny_label = label[0:600, 0:9000]
    tiny_my_label = my_label[0:600, 0:9000]
    tiny_pos_label = pos_label[0:600, 0:9000]

    dataset_len = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            dataset_len += 1

    train_x = np.zeros([dataset_len, mid_hidden_size*2])
    train_y = np.zeros([dataset_len, 1])  # 1--if has building, 0-- if no building
    print("start building......")
    ids = 0
    pos_len = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            image_2015 = np.array(scale_percentile(tiny_2015[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2015 = dae_server(image_2015)
            image_2017 = np.array(scale_percentile(tiny_2017[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2017 = dae_server(image_2017)
            train_x[ids, 0:mid_hidden_size] = np.reshape(hidden_2015, [mid_hidden_size])
            train_x[ids, mid_hidden_size:] = np.reshape(hidden_2017, [mid_hidden_size])
            cur_label = tiny_label[xstart:xend, ystart:yend]
            cur_my_label = tiny_my_label[xstart:xend, ystart:yend]
            cur_pos_label = tiny_pos_label[xstart:xend, ystart:yend]
            if np.sum(cur_label) >= 16*16 and (np.sum(cur_my_label) >= 16*16 or np.sum(cur_pos_label) >= 16*16):
                train_y[ids] = 1
                pos_len += 1
            else:
                train_y[ids] = 0
            ids += 1

    print("train_data size is %d, positive samples are %d." % (ids, pos_len))   # 79591, 335
    dataset = {"train_x": train_x, "train_y": train_y}

    output_file = open("/home/zyt/data/TianChi/dataset/20171030/train_data.pkl", "wb")
    pickle.dump(dataset, output_file)
    output_file.close()

    print("build dataset successfully!")


def build_dataset_3():
    FILE_2015 = '/home/zyt/data/TianChi/20171105_quarterfinals/quarterfinals_2015.tif'
    FILE_2017 = '/home/zyt/data/TianChi/20171105_quarterfinals/quarterfinals_2017.tif'
    FILE_label = '/home/zyt/data/TianChi/label/label1110.tif'

    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)
    label = np.array(tiff.imread(FILE_label), dtype=np.float32)

    image_size = FLAGS.image_size
    channels = FLAGS.channels
    # define in the DAE15_17_32.py
    mid_hidden_size = FLAGS.mid_hidden_size

    # 300:1100, 2200:8900
    tiny_2015 = im_2015[300:1100, 2200:8900, :]
    tiny_2017 = im_2017[300:1100, 2200:8900, :]
    tiny_label = label[300:1100, 2200:8900]

    # 2500:3000, 6500:15000
    tiny_2015_2 = im_2015[2500:3000, 6500:15000, :]
    tiny_2017_2 = im_2017[2500:3000, 6500:15000, :]
    tiny_label_2 = label[2500:3000, 6500:15000]

    dataset_len = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            dataset_len += 1

    train_x = np.zeros([dataset_len, mid_hidden_size*2])
    train_y = np.zeros([dataset_len, 1])  # 1--if has building, 0-- if no building
    print("start building......")
    ids = 0
    pos_len = 0
    for xstart, xend in zip(range(0, tiny_2015.shape[0], 8), range(image_size, tiny_2015.shape[0], 8)):
        for ystart, yend in zip(range(0, tiny_2015.shape[1], 8), range(image_size, tiny_2015.shape[1], 8)):
            image_2015 = np.array(scale_percentile(tiny_2015[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2015 = dae_server(image_2015)
            image_2017 = np.array(scale_percentile(tiny_2017[xstart:xend, ystart:yend, :]), dtype=np.float32)
            hidden_2017 = dae_server(image_2017)
            train_x[ids, 0:mid_hidden_size] = np.reshape(hidden_2015, [mid_hidden_size])
            train_x[ids, mid_hidden_size:] = np.reshape(hidden_2017, [mid_hidden_size])
            cur_label = tiny_label[xstart:xend, ystart:yend]
            if np.sum(cur_label) >= (32*32/2):
                train_y[ids] = 1
                pos_len += 1
            else:
                train_y[ids] = 0
            ids += 1

    print("train_data size is %d, positive samples are %d." % (ids, pos_len))
    dataset = {"train_x": train_x, "train_y": train_y}

    output_file = open("/home/zyt/data/TianChi/dataset/20171110/train_data_ubuntu4.pkl", "wb")
    pickle.dump(dataset, output_file)
    output_file.close()

    print("build dataset successfully!")


def fc_iterator(batch_size=100):
    input_file = open("/home/zyt/data/TianChi/dataset/20170925/train_data.pkl", "rb")
    train_data = pickle.load(input_file)
    input_file.close()
    train_x = train_data['train_x']
    train_y = train_data['train_y']

    epoch_size = len(train_x)//batch_size

    ids = range(epoch_size)
    random.shuffle(ids)

    for i in ids:
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        yield batch_x, batch_y


def fc_iterator_2(train_data, batch_size=100):
    train_x = train_data['train_x']
    train_y = train_data['train_y']

    epoch_size = len(train_x)//batch_size

    ids = range(epoch_size)
    random.shuffle(ids)

    for i in ids:
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        yield batch_x, batch_y


def whole_pic_iterator(im_2015, im_2017, label, batch_size=100):
    w, h, d = im_2015.shape
    x_ids = range(0, w-FLAGS.image_size, 16)
    y_ids = range(0, h-FLAGS.image_size, 16)

    lens = (len(x_ids)/10) * (len(y_ids)/10)
    print("whole_batch_len: %d" % lens)

    random.shuffle(x_ids)
    random.shuffle(y_ids)

    batch_len = 0
    batch_x = np.zeros([batch_size, 2*FLAGS.mid_hidden_size])
    batch_y = np.zeros([batch_size, 1])
    for x in x_ids:
        for y in y_ids:
            cur_2015 = np.array(scale_percentile(im_2015[x:x+FLAGS.image_size, y:y+FLAGS.image_size, :]), dtype=np.float32)
            cur_2017 = np.array(scale_percentile(im_2017[x:x+FLAGS.image_size, y:y+FLAGS.image_size, :]), dtype=np.float32)
            hidden_2015 = dae_server(cur_2015)
            hidden_2017 = dae_server(cur_2017)
            batch_x[batch_len, 0:FLAGS.mid_hidden_size] = hidden_2015
            batch_x[batch_len, FLAGS.mid_hidden_size:] = hidden_2017
            cur_label = label[x:x+FLAGS.image_size, y:y+FLAGS.image_size]
            if np.sum(cur_label) >= (32*32/2):
                batch_y[batch_len] = 1
            batch_len += 1
            if batch_len >= 100:
                batch_len = 0
                yield batch_x, batch_y
                batch_x = np.zeros([batch_size, 2*FLAGS.mid_hidden_size])
                batch_y = np.zeros([batch_size, 1])


def whole_pic_iterator2(im_2015, im_2017, label, batch_size=64):
    w, h, d = im_2015.shape
    x_ids = range(0, w-FLAGS.image_size, 16)
    y_ids = range(0, h-FLAGS.image_size, 16)

    lens = (len(x_ids)/8) * (len(y_ids)/8)
    print("whole_batch_len: %d" % lens)

    random.shuffle(x_ids)
    random.shuffle(y_ids)

    batch_len = 0
    cur_2015 = np.zeros([batch_size, FLAGS.image_size, FLAGS.image_size, 4])
    cur_2017 = np.zeros([batch_size, FLAGS.image_size, FLAGS.image_size, 4])
    batch_x = np.zeros([batch_size, 2*FLAGS.mid_hidden_size])
    batch_y = np.zeros([batch_size, 1])

    tmp_graph = tf.Graph()
    with tmp_graph.as_default():

        ckpt = tf.train.get_checkpoint_state("/home/zyt/data/TianChi/DAE/log/log15_17_32/")
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            for x in x_ids:
                for y in y_ids:
                    cur_2015[batch_len] = np.array(scale_percentile(im_2015[x:x+FLAGS.image_size, y:y+FLAGS.image_size, :]), dtype=np.float32)
                    cur_2017[batch_len] = np.array(scale_percentile(im_2017[x:x+FLAGS.image_size, y:y+FLAGS.image_size, :]), dtype=np.float32)
                    cur_label = label[x:x+FLAGS.image_size, y:y+FLAGS.image_size]
                    if np.sum(cur_label) >= (32*32/2):
                        batch_y[batch_len] = 1
                    batch_len += 1
                    if batch_len >= 64:
                        batch_len = 0
                        hidden_2015 = sess.run(tf.get_default_graph().get_tensor_by_name('max_pooling2d_3/MaxPool:0'), feed_dict={'inputs:0': cur_2015})
                        hidden_2017 = sess.run(tf.get_default_graph().get_tensor_by_name('max_pooling2d_3/MaxPool:0'), feed_dict={'inputs:0': cur_2017})
                        batch_x[:, 0:FLAGS.mid_hidden_size] = np.reshape(hidden_2015, [batch_size, -1])
                        batch_x[:, FLAGS.mid_hidden_size:] = np.reshape(hidden_2017, [batch_size, -1])
                        yield batch_x, batch_y
                        cur_2015 = np.zeros([batch_size, FLAGS.image_size, FLAGS.image_size, 4])
                        cur_2017 = np.zeros([batch_size, FLAGS.image_size, FLAGS.image_size, 4])
                        batch_x = np.zeros([batch_size, 2*FLAGS.mid_hidden_size])
                        batch_y = np.zeros([batch_size, 1])


def main(_):
    #build_dataset_3()

    FILE_2015 = '/home/zyt/data/TianChi/20171105_quarterfinals/quarterfinals_2015.tif'
    FILE_2017 = '/home/zyt/data/TianChi/20171105_quarterfinals/quarterfinals_2017.tif'
    FILE_label = '/home/zyt/data/TianChi/label/label1110.tif'

    im_2015 = np.array(tiff.imread(FILE_2015).transpose([1, 2, 0]), dtype=np.float32)
    im_2017 = np.array(tiff.imread(FILE_2017).transpose([1, 2, 0]), dtype=np.float32)
    label = np.array(tiff.imread(FILE_label), dtype=np.float32)
    x, y = whole_pic_iterator2(im_2015, im_2017, label)
    print(x.shape)
    print(y.shape)
    print(np.max(x))
    print(np.max(y))

if __name__ == '__main__':
    tf.app.run()
