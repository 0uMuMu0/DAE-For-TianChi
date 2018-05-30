import tensorflow as tf
import numpy as np
import produce_images_by_dae
import os
import pickle

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.client import device_lib

flags = tf.flags

flags.DEFINE_integer('hidden_size', 4*4*64, 'the hidden size of image(dae)')
flags.DEFINE_string("model_path", "tmp/FC_16_2/",
                    "SaveModel path.")
flags.DEFINE_string("model_version", 1,
                    "the version of model")

FLAGS = flags.FLAGS


def train(batch_size=100, lr=0.00005, max_epoches=200):

    x = tf.placeholder(tf.float32, shape=[None, FLAGS.hidden_size*2])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    w1 = tf.Variable(tf.truncated_normal(shape=[FLAGS.mid_hidden_size*2, 200], stddev=0.5), name="weight1")
    b1 = tf.Variable(tf.zeros(shape=[200]), name="bias1")
    layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal(shape=[200, 1], stddev=0.5), name="weight2")
    b2 = tf.Variable(tf.zeros(shape=[1]), name="bias2")
    layer2 = tf.matmul(layer1, w2) + b2

    pred = tf.sigmoid(layer2)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=layer2, pos_weight=50))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct = (np.abs(y-pred) < 0.1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init_op = tf.global_variables_initializer()

    input_file = open("/home/zyt/data/TianChi/dataset/20171024/train_data.pkl", "rb")
    train_data = pickle.load(input_file)
    input_file.close()

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(max_epoches):
            total_loss = 0
            total_accuracy = 0
            for step, (batch_x, batch_y) in enumerate(produce_images_by_dae.fc_iterator_2(train_data, 100)):
                _, l, accu = sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y: batch_y})

                total_loss += l
                avg_loss = total_loss/(step+1)
                total_accuracy += accu
                avg_accu = total_accuracy/(step+1)
                if step % 50 == 0:
                    print("Epoch %d: at step %d, average loss is %f, accuracy is %f." % (epoch, step, avg_loss, avg_accu))

        # Export tensorflow serving
        export_path = os.path.join(tf.compat.as_bytes(FLAGS.model_path),
                                   tf.compat.as_bytes(str(FLAGS.model_version)))
        builder = saved_model_builder.SavedModelBuilder(export_path)
        prediction_inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
        prediction_outputs = {'output': tf.saved_model.utils.build_tensor_info(pred)}
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=prediction_inputs,
            outputs=prediction_outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                'predict_signature': prediction_signature,
                                             })
        builder.save()
        print("Done export!")


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
