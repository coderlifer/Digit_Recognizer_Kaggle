# Digit_Recognizer_Kaggle

Tensorflow implemention of CNN for Kaggle competition: [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/overview).


## Result

99.175% is achieved after 100k training steps. Better results can be achieved by tuning hyperparameters.


## How to run

1. Split train and validation dataset.

    ```shell
    python help.py
    ```

2. Training, validation and inference.

    Modify the [code](http://git.oschina.net/DavisWade/digit_recognizer/blob/master/train.py?dir=0&filepath=train.py&oid=c73ade663356bf05fbff5cc8dcdac5fe4839f1b5&sha=643e20e7e07b08df62c395baaaf4bee39444483b#L231) bellow based your own setting.

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

