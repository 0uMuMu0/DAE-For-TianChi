# coding=utf-8
import tensorflow as tf
import numpy as np
import input_data_minst as input_data

flags = tf.flags

flags.DEFINE_integer('max_epoch', 100, '')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('lr', 0.02, 'learning rate')
flags.DEFINE_integer('image_size', 28, 'picture size')
flags.DEFINE_integer('n_hidden', 500, 'hidden size')
flags.DEFINE_integer('corruption_level', 0.3, '')

FLAGS = flags.FLAGS


image_size = FLAGS.image_size
n_visible = image_size * image_size
n_hidden = FLAGS.n_hidden
corruption_level = FLAGS.corruption_level

# 输入的minst图片大小是28*28
x = tf.placeholder("float", [None, n_visible], name='x_2015')

# 用于将部分输入数据置为0
mask = tf.placeholder("float", [None, n_visible], name="mask")

W_init_max = 4*np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden], minval=-W_init_max, maxval=W_init_max)

# encoder
w = tf.Variable(W_init, name='weights')
b = tf.Variable(tf.zeros(shape=[n_hidden]), name='bias')

# decoder
w_prime = tf.transpose(w)
b_prime = tf.Variable(tf.zeros(shape=[n_visible]), name='bias_prime')


def model(x, mask, w, b, w_prime, b_prime):
    corrupted_x = mask * x
    y = tf.nn.sigmoid(tf.matmul(corrupted_x, w) + b)  # hidden state
    z = tf.nn.sigmoid(tf.matmul(y, w_prime) + b_prime)  # reconstructed input
    return z


# build model graph
z = model(x, mask, w, b, w_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.square(x - z))
train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(cost)

# load dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(FLAGS.max_epoch):
        for start, end in zip(range(0, len(train_x)-1, FLAGS.batch_size), range(FLAGS.batch_size, len(train_x), FLAGS.batch_size)):
            batch_x = train_x[start:end]
            mask_np = np.random.binomial(1, 1-FLAGS.corruption_level, batch_x.shape)
            sess.run(train_op, feed_dict={x: batch_x, mask: mask_np})

        mask_np = np.random.binomial(1, 1-FLAGS.corruption_level, test_x.shape)
        print(epoch, sess.run(cost, feed_dict={x: test_x, mask: mask_np}))




