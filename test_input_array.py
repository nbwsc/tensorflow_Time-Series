# coding:utf-8

"""测试公式生成序列"""

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
#print(x, noise)

y = np.sin(np.pi * x / 100) + x / 200. + noise
plt.plot(x, y)
# plt.show()
plt.savefig('timeseries_y.jpg')

data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y
}
# data = {‘times’:x, ‘values’:y}

reader = NumpyReader(data)

# tf read data
# 不能直接使用sess.run(reader.read_full())来从reader中取出
# 所有数据。原因在于read_full()方法会产生读取队列，而队列的线程
# 此时还没启动，我们需要使用tf.train.start_queue_runners
# 启动队列，才能使用sess.run()来获取值。
with tf.Session() as sess:
    full_data = reader.read_full()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(full_data))
    coord.request_stop()

# tf.contrib.timeseries.RandomWindowInputFn会在reader的
# 所有数据中，随机选取窗口长度为window_size的序列，并包装成
# batch_size大小的batch数据。换句话说，一个batch内共有
# batch_size个序列，每个序列的长度为window_size。
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=2, window_size=10)

with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch = sess.run(batch_data[0])
    coord.request_stop()


# plt.figure
# plt.plot(one_batch)
# plt.show()
print('one_batch data:', one_batch)
plt.figure
plt.plot(one_batch['times'], one_batch['values'])
