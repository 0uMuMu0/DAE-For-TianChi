import tensorflow as tf
import numpy as np
import produce_data
import matplotlib.pyplot as plt

from PIL import Image
import tifffile as tiff


def train(batch_size=100, lr=0.00005, max_epochs=30):

    x = tf.placeholder(tf.float32, shape=[None, 4])
    y = tf.placeholder(tf.float32, shape=[None, 2])

    #w = tf.Variable(tf.truncated_normal(shape=(4, 2), stddev=0.5), name="weight")
    w1 = tf.Variable(tf.zeros(shape=[4, 10]), name="weight1")
    #w = tf.Variable(tf.random_uniform(shape=[4, 2], minval=-0.001, maxval=0.001), name="weight")
    b1 = tf.Variable(tf.zeros(shape=[10]), name="bias1")
    layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.zeros(shape=[10, 2]), name="weight2")
    b2 = tf.Variable(tf.zeros(shape=[2]), name="bias2")

    pred = tf.nn.softmax(tf.matmul(layer1, w2) + b2)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    loss = tf.reduce_mean(tf.square(y-pred))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init_op = tf.global_variables_initializer()

    train_x, train_y = produce_data.bulid_dataset(600, 800, 10450, 10650)
    train_x2, train_y2 = produce_data.bulid_dataset(800, 1000, 4950, 5150)
    train_x3, train_y3 = produce_data.bulid_dataset(1220, 1420, 8250, 8450)
    train_x4, train_y4 = produce_data.bulid_dataset(4250, 4450, 2000, 2200)
    test_x, test_y = produce_data.bulid_dataset(1000, 1500, 8000, 8500)
    #all_data_x, all_data_y = produce_data.bulid_dataset(0, 5106, 0, 5000)
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(max_epochs):
            avg_loss = 0
            total_batch = len(train_x)/batch_size
            for i in range(total_batch):
                _, curloss = sess.run([optimizer, loss],
                                      feed_dict={x: train_x[i*batch_size:(i+1)*batch_size, :],
                                                 y: train_y[i*batch_size:(i+1)*batch_size, :]})
                avg_loss += curloss

            avg_loss = avg_loss/total_batch
            print("Epoch", epoch, ": average loss=", avg_loss)
            res = pred.eval({x: train_x, y: train_y})

            res = np.argmax(res, axis=1)
            res = np.reshape(res, [200, 200])
            data = np.argmax(train_y, axis=1)
            data = np.reshape(data, [200, 200])

            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.imshow(res)
            ax = fig.add_subplot(222)
            ax.imshow(data)

            res_t = pred.eval({x: train_x2, y: train_y2})

            res_t = np.argmax(res_t, axis=1)
            res_t = np.reshape(res_t, [200, 200])
            data2 = np.argmax(train_y2, axis=1)
            data2 = np.reshape(data2, [200, 200])

            ax = fig.add_subplot(223)
            ax.imshow(res_t)
            ax = fig.add_subplot(224)
            ax.imshow(data2)

            plt.show()
        print("Testing Accuracy: ", accuracy.eval({x: test_x, y: test_y}))

        """all_data_pred = pred.eval({x: all_data_x, y: all_data_y})
        print("start to save image...")
        image = np.argmax(all_data_pred, axis=1)
        out_array = np.reshape(image, [5106, 5000])
        out_array = np.array(out_array, dtype=np.uint8)
        tiff.imsave('tmp.tif', out_array)
        #img_out = Image.fromarray(out_array)
        #img_out.save("result2.tif")
        print("save successfully!")"""


if __name__ == '__main__':
    train()
