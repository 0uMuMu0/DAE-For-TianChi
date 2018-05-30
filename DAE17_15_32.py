# coding=utf-8
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data_dae

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.client import device_lib


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer('max_epoch', 10, '')
flags.DEFINE_integer('size', 64, 'batch size')
flags.DEFINE_integer('lr', 0.0005, 'learning rate')
flags.DEFINE_integer('image_size', 32, 'picture size')
flags.DEFINE_integer('channels', 4, '')
flags.DEFINE_integer('noise_factor', 0.2, '')
flags.DEFINE_string("model_path", "/tmp/DAE17_15_32/",
                    "SaveModel path.")
flags.DEFINE_string("model_version", 1,
                    "the version of model")
flags.DEFINE_string("save_path", "log_17_15_32/",
                    "Model output directory.")
FLAGS = flags.FLAGS


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 4))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def get_compare_image(inputs, output, label, nums):
    h, w = inputs.shape[1], inputs.shape[2]
    images = np.zeros((h*4, w*nums, 4))
    for i in range(nums):
        images[0:h, i*w:(i+1)*w, :] = inputs[i]
    for i in range(nums):
        images[1*h:2*h, i*w:(i+1)*w, :] = output[i]
    for i in range(nums):
        images[2*h:3*h, i*w:(i+1)*w, :] = label[i]
    return images


# Input data, 输入的图片大小是32×32
x = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels], name='inputs')  # inputs
y = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels], name='targets')  # targets
# 加噪声，将部分输入变为0
mask = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels], name="mask")

# Model
conv1 = tf.layers.conv2d(x, 64, (5, 5), padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

conv2 = tf.layers.conv2d(pool1, 64, (5, 5), padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

conv3 = tf.layers.conv2d(pool2, 64, (5, 5), padding='same', activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')

conv4_resize = tf.image.resize_nearest_neighbor(pool3, (8, 8))
conv5 = tf.layers.conv2d(conv4_resize, 64, (5, 5), padding='same', activation=tf.nn.relu)

conv5_resize = tf.image.resize_nearest_neighbor(conv5, (16, 16))
conv6 = tf.layers.conv2d(conv5_resize, 64, (5, 5), padding='same', activation=tf.nn.relu)

conv6_resize = tf.image.resize_nearest_neighbor(conv6, (32, 32))
conv7 = tf.layers.conv2d(conv6_resize, 64, (5, 5), padding='same', activation=tf.nn.relu)

y_conv = tf.layers.conv2d(conv7, FLAGS.channels, (5, 5), padding='same', activation=None)

outputs = tf.nn.sigmoid(y_conv)

# create cost function
loss = tf.reduce_mean(tf.nn.l2_loss(y_conv - y))
optimizer = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

sv = tf.train.Supervisor(logdir=FLAGS.save_path)
config_proto = tf.ConfigProto(allow_soft_placement=False)
test_2015, test_2017 = input_data_dae.test_data_32()
with sv.managed_session(config=config_proto) as sess:
    plt.ion()
    for epoch in range(FLAGS.max_epoch):
        for step, (batch_x, batch_y)in enumerate(input_data_dae.dae_iterator_32(FLAGS.size)):
            # noisy_x = batch_x + np.random.randn(*batch_x.shape)
            # noisy_x = np.clip(noisy_x, 0.0, 1.0)

            _, y, l = sess.run([optimizer, y_conv, loss], feed_dict={x: batch_y, y: batch_x})

            if step % 100 == 0:
                print("Epoch %d: at step %d, loss is %f" % (epoch, step, l))
                plt.clf()
                plt.imshow(get_compare_image(batch_x[5:10], y[5:10], batch_y[5:10], 5))
                # plt.imshow(merge(y, [8, 8]))
                plt.text(-2.0, -5.0, "Epoch %d: at step %d, loss is %f" % (epoch, step, l), fontdict={'size': 10})
                plt.draw()
                plt.pause(0.1)

        if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(sess, FLAGS.save_path, global_step=sv.global_step)
            print("Save successfully!")

        test_loss = sess.run(loss, feed_dict={x: test_2017, y: test_2015})
        print("Epoch %d: test loss is %f" % (epoch, test_loss))

    plt.ioff()
    plt.show()

    sess.graph._unsafe_unfinalize()

    # Export tensorflow serving
    export_path = os.path.join(tf.compat.as_bytes(FLAGS.model_path),
                               tf.compat.as_bytes(str(FLAGS.model_version)))
    builder = saved_model_builder.SavedModelBuilder(export_path)
    prediction_inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
    prediction_outputs = {'output': tf.saved_model.utils.build_tensor_info(y_conv),
                          'mid_hidden': tf.saved_model.utils.build_tensor_info(pool3)}
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=prediction_inputs,
        outputs=prediction_outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                            'predict_signature': prediction_signature,
                                         })
    sess.graph.finalize()
    builder.save()
    print("Done export!")
