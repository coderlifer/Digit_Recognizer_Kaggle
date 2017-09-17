import numpy as np
import tensorflow as tf


def shuffle_split(data_param, validation_size_param):
    shuffle_indices = np.random.permutation(np.arange(data_param.shape[0]))
    data_param = data_param[shuffle_indices]

    val_indices = np.random.choice(data_param.shape[0], validation_size_param, replace=False)
    data_val = data_param[val_indices]
    X_val_ = data_val[:, 1:]
    y_val_ = data_val[:, 0]
    y_val_ = np.reshape(y_val_, [-1, 1])

    train_indices = [i for i in xrange(data_param.shape[0]) if i not in val_indices]
    data_train = data_param[train_indices]
    X_train_ = data_train[:, 1:]
    y_train_ = data_train[:, 0]
    y_train_ = np.reshape(y_train_, [-1, 1])

    return X_val_, y_val_, X_train_, y_train_


def split_save(dp, vs):
    data = np.genfromtxt(dp, delimiter=',', skip_header=1)
    data = data.astype(np.float32)

    X_v, y_v, X_t, y_t = shuffle_split(data, vs)
    print(X_t.shape)
    print(X_t)
    print(y_t.shape)
    print(y_t)
    np.savetxt('./DigitRecognizer/X_validation.csv', X_v, fmt='%s', delimiter=',')
    np.savetxt('./DigitRecognizer/y_validation.csv', y_v, fmt='%s', delimiter=',')
    np.savetxt('./DigitRecognizer/X_train.csv', X_t, fmt='%s', delimiter=',')
    np.savetxt('./DigitRecognizer/y_train.csv', y_t, fmt='%s', delimiter=',')


def load_display(path_list):
    input_queue = tf.train.string_input_producer(path_list)

    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(input_queue)
    val = tf.decode_csv(records=value,
                        record_defaults=[0])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        v = sess.run(val)
        print(v.shape)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    data_path = './DigitRecognizer/train.csv'
    validation_size = 8000  # 34000

    split_save(data_path, validation_size)


# python ./train.py  --batch_count=80  --num_gpus=0
