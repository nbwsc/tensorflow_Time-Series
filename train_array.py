# coding:utf-8

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader


def main(_):
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 100) + x / 200.
    plt.plot(x, y)
    plt.savefig('timeseries_y.jpg')

    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
    }

    reader = NumpyReader(data)

    """
        tf.contrib.timeseries.RandomWindowInputFn会在reader的所有数据中，
        随机选取窗口长度为window_size的序列，并包装成batch_size大小的batch数据。
        换句话说，一个batch内共有batch_size个序列，每个序列的长度为window_size
    """
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=16, window_size=40)

    """
        ARRegressor:
        第一个参数periodicities表示序列的规律性周期。
        我们在定义数据时使用的语句是：“y = np.sin(np.pi * x / 100) + x / 200. + noise”，
        因此周期为200。
        input_window_size表示模型每次输入的值，
        output_window_size表示模型每次输出的值。
        input_window_size和output_window_size加起来必须等于train_input_fn中总的window_size。
        在这里，我们总的window_size为40，input_window_size为30，
        output_window_size为10，也就是说，一个batch内每个序列的长度为40，
        其中前30个数被当作模型的输入值，后面10个数为这些输入对应的目标输出值。
        最后一个参数loss指定采取哪一种损失，一共有两
        种损失可以选择，分别是NORMAL_LIKELIHOOD_LOSS和SQUARED_LOSS。

        num_features参数表示在一个时间点上观察到的数的维度。我们这里每一步都是一个单独的值，所以num_features=1。

        还有一个比较重要的参数是model_dir。它表示模型训练好后保存的地址，如果不指定的话，就会随机分配一个临时地址。
    """
    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=200, input_window_size=30, output_window_size=10,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.SQUARED_LOSS)

    ar.train(input_fn=train_input_fn, steps=6000)

    """
        TFTS中验证(evaluation)的含义是：使用训练好的模型在原先的训练集上进行计算，由此我们可以观察到模型的拟合效果
        如果要理解这里的逻辑，首先要理解之前定义的AR模型：
        它每次都接收一个长度为30的输入观测序列，并输出长度为10的预测序列。
        整个训练集是一个长度为1000的序列，前30个数首先被当作“初始观测序列”输入到模型中，由此就可以计算出下面10步的预测值。
        接着又会取30个数进行预测，这30个数中有10个数就是前一步的预测值，新得到的预测值又会变成下一步的输入，以此类推。

        最终我们得到970个预测值（970=1000-30，因为前30个数是没办法进行预测的）。
        这970个预测值就被记录在evaluation[‘mean’]中。evaluation还有其他几个键值，
        如evaluation[‘loss’]表示总的损失，evaluation[‘times’]表示evaluation[‘mean’]对应的时间点等等。
    """
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed',
    # 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=250)))

    print('loss', evaluation['loss'])

    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1),
             data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1),
             evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1),
             predictions['mean'].reshape(-1), label='prediction')

    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result.jpg')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
