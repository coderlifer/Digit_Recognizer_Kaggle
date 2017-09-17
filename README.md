# Digit_Recognizer_Kaggle

Tensorflow implemention of a CNN model for Kaggle competition: [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/overview).


## Result

99.175% is achieved after 100k training steps. Better results can be achieved by tuning hyperparameters.


## How to run

1. Split train and validation dataset.

    ```shell
    python help.py
    ```

2. Training, validation and inference.

    Modify the [code](https://github.com/watsonyanghx/Digit_Recognizer_Kaggle/blob/master/train.py#L230) bellow based your own setting.

    ```shell
    hps = net.HParams(data_size=34000,

                      batch_size=100,

                      num_classes=10,

                      lrn_rate=0.01,

                      mode='validation',

                      weight_decay_rate=0.04)
    ```

    then run:

    ```shell
    python ./train.py  --batch_count=80  --num_gpus=1
    ```

    `batch_count` is the numbers of batch your are going to run in `validation` or `infer` mode, basically: `batch_count = hps.data_size / hps.batch_size`.

