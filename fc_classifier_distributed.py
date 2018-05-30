# coding=utf-8

import tensorflow as tf
import numpy as np
import produce_images_by_dae
import os
import pickle
import time

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.client import device_lib

flags = tf.flags

flags.DEFINE_integer('hidden_size', 4*4*64, 'the hidden size of image(dae)')
flags.DEFINE_string("data_path", "/home/zyt/data/TianChi/dataset/20170922/train_data.pkl", "the directory for train data")
flags.DEFINE_string("model_path", "tmp/FC_disrtributed/",
                    "SaveModel path.")
flags.DEFINE_string("model_version", 1,
                    "the version of model")
flags.DEFINE_string("save_path", "log/log_FC/",
                    "Model output directory.")
flags.DEFINE_integer("lr", 0.00005, "learning rate")
flags.DEFINE_integer("max_epoch", 150, "")
flags.DEFINE_integer("steps_of_one_epoch", 850, "")

# 指定当前运行的是参数服务器还是计算服务器
flags.DEFINE_string('job_name', '', ' "ps" or "worker" ')
# 指定集群中的参数服务器地址
flags.DEFINE_string('ps_hosts', 'zyt-HP:2223',
                    'Comma-separated list of hostname:port for the parameter server jobs.')
# 指定集群中的计算服务器地址
flags.DEFINE_string('worker_hosts', 'zyt-HP:2222, ubuntu1:2222,ubuntu2:2222,ubuntu4:2222',
                    'Comma-separated list of hostname:port for the worker jobs.')
# 指定当前程序的任务index
flags.DEFINE_integer('task_index', 0, 'Task ID of the worker/replica running the training.')

FLAGS = flags.FLAGS


def main(_):
    # 解析flags并通过tf.train.ClusterSpec配置Tensorflow集群
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 通过ClusterSpec以及当前任务创建Server
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # 参数服务器只需要管理Tensorflow中的变量，不需要执行训练的过程。server.join()会一直停在这条语句上
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        # worker需要定义计算服务器需要运行的操作。在所有的计算服务器中有一个是主计算服务器，它除了负责计算反向传播的结果，还负责输出日志和保存模型。
        input_file = open(FLAGS.data_path, "rb")
        train_data = pickle.load(input_file)
        input_file.close()

        # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备
        # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而计算分配到当前的计算服务器上
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            x = tf.placeholder(tf.float32, shape=[None, FLAGS.hidden_size*2])
            y = tf.placeholder(tf.float32, shape=[None, 1])

            w1 = tf.Variable(tf.truncated_normal(shape=[FLAGS.mid_hidden_size*2, 200], stddev=0.5), name="weight1")
            b1 = tf.Variable(tf.zeros(shape=[200]), name="bias1")
            layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

            w2 = tf.Variable(tf.truncated_normal(shape=[200, 1], stddev=0.5), name="weight2")
            b2 = tf.Variable(tf.zeros(shape=[1]), name="bias2")
            layer2 = tf.matmul(layer1, w2) + b2

            pred = tf.sigmoid(layer2)

            loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=layer2, pos_weight=100))
            tf.summary.scalar("loss", loss)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss, global_step=global_step)

            correct = (np.abs(y-pred) < 0.1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            # tf.train.Supervisor能统一管理队列操作，模型保存，日志输出以及会话的生成
            # is_chief定义当前计算服务器是否为主计算服务器，只有主计算服务器会保存模型以及输出日志
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir=FLAGS.save_path, init_op=init_op,
                                     summary_op=summary_op, saver=saver, save_model_secs=600, global_step=global_step)
            config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            # 通过Supervisor生成会话
            sess = sv.prepare_or_wait_for_session(server.target, config=config_proto)
            max_train_steps = FLAGS.max_epoch * FLAGS.steps_of_one_epoch
            step = 0          # global
            epoch = 0

            if sv.is_chief:
                while step < max_train_steps:
                    step = sess.run(global_step)
                    print("time:%f, step is %d." % (time.time(), step))
                    print(sv.should_stop())
                    time.sleep(60)
                # Test
                test_loss = 0
                test_accu = 0
                avg_test_loss = 0
                avg_test_accu = 0
                for batch_id, (batch_x, batch_y) in enumerate(produce_images_by_dae.fc_iterator_2(train_data, 100)):
                    l, accu = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
                    test_loss += l
                    test_accu += accu
                    avg_test_loss = test_loss/(batch_id+1)
                    avg_test_accu = test_accu/(batch_id+1)
                print("Test: average loss is %f, accuracy is %f." % (avg_test_loss, avg_test_accu))
            else:
                while not sv.should_stop() and step < max_train_steps:
                    try:
                        total_loss = 0
                        total_accuracy = 0
                        for batch_id, (batch_x, batch_y) in enumerate(produce_images_by_dae.fc_iterator_2(train_data, 100)):
                            _, l, accu, step = sess.run([optimizer, loss, accuracy, global_step], feed_dict={x: batch_x, y: batch_y})

                            total_loss += l
                            avg_loss = total_loss/(batch_id+1)
                            total_accuracy += accu
                            avg_accu = total_accuracy/(batch_id+1)
                            if batch_id % 50 == 0:
                                print("Epoch %d: at batch %d(global_step %d) , average loss is %f, accuracy is %f." % (epoch, batch_id, step, avg_loss, avg_accu))
                        epoch += 1
                    except Exception as ex:
                        print("wrong:%s" % ex.message)

            if sv.is_chief:
                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(sess, FLAGS.save_path, global_step=sv.global_step)
                    print("Save successfully!")

                # Export tensorflow serving
                sess.graph._unsafe_unfinalize()
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

                legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                     signature_def_map={
                                                     'predict_signature': prediction_signature,
                                                     },
                                                     clear_devices=True)
                sess.graph.finalize()
                builder.save()
                print("Done export!")
            sv.stop()


if __name__ == '__main__':
    tf.app.run()
