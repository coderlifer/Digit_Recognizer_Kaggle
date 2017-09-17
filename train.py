import numpy as np
import tensorflow as tf
import math
import time

import net


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './tmp/model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('log_root', './tmp/model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('eval_dir', './tmp/model/validation',
                           'Directory to keep validation outputs.')
tf.app.flags.DEFINE_integer('batch_count', 80,
                            'Directory to keep validation outputs.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')


def train_loop(hps, X_train, y_train):
    num_batches = int(math.ceil(hps.data_size / float(hps.batch_size)))
    global_steps = 0

    model = net.Net(hps)
    model.build_graph()

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    # 'Saver' op to save and restore all the variables
    # saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     for _ in hps.max_number_of_steps:
    #         if global_steps % num_batches:
    #             start = hps.batch_size * global_steps
    #             end = np.minimum(hps.data_size, hps.batch_size * (global_steps + 1))
    #
    #             X = X_train[start:end]
    #             y = y_train[start:end]
    #             fd = {model.X: X,
    #                   model.y: y}
    #
    #             _, global_steps, loss, acc = sess.run([model.train_op,
    #                                                    model.cost,
    #                                                    precision],
    #                                                   feed_dict=fd)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.01

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 2000:
                self._lrn_rate = 0.01
            elif train_step < 6000:
                self._lrn_rate = 0.008
            elif train_step < 20000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0002

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            save_checkpoint_secs=60,
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:

        while not mon_sess.should_stop():
            curr_batch_start = global_steps % num_batches
            global_steps += 1

            if curr_batch_start == 0:
                shuffle_indices = np.random.permutation(np.arange(X_train.shape[0]))
                X_train = X_train[shuffle_indices]
                y_train = y_train[shuffle_indices]

            start = hps.batch_size * curr_batch_start
            end = np.minimum(hps.data_size, hps.batch_size * (curr_batch_start + 1))

            xx = X_train[start:end]
            yy = y_train[start:end]
            fd = {model.X: xx,
                  model.y: yy}

            mon_sess.run(model.train_op, feed_dict=fd)


def evaluate(hps, X_val, y_val):
    model = net.Net(hps)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for i in xrange(FLAGS.batch_count):
            start = hps.batch_size * i
            end = hps.batch_size * (i + 1)

            xx = X_val[start:end]
            yy = y_val[start:end]
            fd = {model.X: xx,
                  model.y: yy}

            summaries, loss, predictions, truth, train_step = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step],
                feed_dict=fd)

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        accuracy = 1.0 * correct_prediction / total_prediction

        precision_summ = tf.Summary()
        precision_summ.value.add(tag='Accuracy', simple_value=accuracy)
        summary_writer.add_summary(precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f' % (loss, accuracy))
        summary_writer.flush()

        time.sleep(10)


def infer(hps, X_infer, y_infer):
    num_batchs = int(math.ceil(hps.data_size / float(hps.batch_size)))
    global_steps = 1

    model = net.Net(hps)
    model.build_graph()

    saver = tf.train.Saver()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    try:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

    saver.restore(sess, ckpt_state.model_checkpoint_path)

    for i in xrange(FLAGS.batch_count):
        start = hps.batch_size * i
        end = hps.batch_size * (i + 1)

        xx = X_infer[start:end]
        yy = y_infer[start:end]
        fd = {model.X: xx,
              model.y: yy}

        pred = sess.run(model.predictions, feed_dict=fd)
        pred = np.argmax(pred, axis=1)

        with open('./submission.csv', 'a') as f:
            pred_list = pred.tolist()
            for j in xrange(len(pred_list)):
                f.write(str(global_steps) + ',' + str(pred_list[j]) + '\n')
                global_steps += 1


def load_data(dp, skip_header=0):
    data = np.genfromtxt(dp, delimiter=',', skip_header=skip_header)
    data = data.astype(np.float32)

    return data


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    # Change values bellow based on your own setting.
    hps = net.HParams(data_size=34000,
                      batch_size=100,
                      num_classes=10,
                      lrn_rate=0.01,
                      mode='validation',
                      weight_decay_rate=0.04)

    with tf.device(dev):
        if hps.mode == 'train':
            # load data
            X_t = load_data('./DigitRecognizer/X_train.csv')
            y_t = load_data('./DigitRecognizer/y_train.csv')
            y_t = np.reshape(y_t, [-1, 1])
            print(X_t.shape)
            print(y_t.shape)

            train_loop(hps, X_t, y_t)
        elif hps.mode == 'validation':
            # load data
            X_v = load_data('./DigitRecognizer/X_validation.csv')
            y_v = load_data('./DigitRecognizer/y_validation.csv')
            y_v = np.reshape(y_v, [-1, 1])
            print(X_v.shape)
            print(y_v.shape)

            evaluate(hps, X_v, y_v)
        elif hps.mode == 'infer':
            # load data
            X_test = load_data('./DigitRecognizer/test.csv', 1)
            y_test = np.reshape(np.arange(X_test.shape[0]), [-1, 1])
            print(X_test.shape)
            print(y_test.shape)

            infer(hps, X_test, y_test)

    # load_display(['/home/yang/Downloads/FILE/CODE/kaggle/DigitRecognizer/train.csv'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
