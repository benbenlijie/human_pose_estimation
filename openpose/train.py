import sys
import os
import pandas
import re
import math
sys.path.append("..")
from openpose.model import get_training_model
from openpose.optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.utils.data_utils import get_file
from keras.applications.vgg19 import VGG19
from openpose.ds_iterator import DataIterator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/data/data/")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", "-lr", type=float, default=4e-3)
parser.add_argument("--max_iter", "-mi", type=int, default=20000)
parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
parser.add_argument("--output_dir", type=str, default="/data/output/")
parser.add_argument("--log_dir", type=str, default="/data/output/")

FLAGS, _ = parser.parse_known_args()

batch_size = FLAGS.batch_size
base_lr = FLAGS.learning_rate
momentum = 0.9
weight_decay = FLAGS.weight_decay
lr_policy =  "step"
gamma = 0.333
stepsize = 5000 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = FLAGS.max_iter #200#000 # 600000
data_shape = [512, 512]

WEIGHTS_BEST = FLAGS.output_dir + "weights.best.h5"
TRAINING_LOG = "training.csv"
LOGS_DIR = FLAGS.log_dir


def get_last_epoch():
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)

model = get_training_model(weight_decay)

from_vgg = dict()
from_vgg['conv1_1'] = 'block1_conv1'
from_vgg['conv1_2'] = 'block1_conv2'
from_vgg['conv2_1'] = 'block2_conv1'
from_vgg['conv2_2'] = 'block2_conv2'
from_vgg['conv3_1'] = 'block3_conv1'
from_vgg['conv3_2'] = 'block3_conv2'
from_vgg['conv3_3'] = 'block3_conv3'
from_vgg['conv3_4'] = 'block3_conv4'
from_vgg['conv4_1'] = 'block4_conv1'
from_vgg['conv4_2'] = 'block4_conv2'

# load previous weights or vgg19 if this is first run
if os.path.exists(WEIGHTS_BEST):
    print("load the best weights...")
    model.load_weights(WEIGHTS_BEST)
    last_epoch = get_last_epoch() + 1
else:
    print("loading vgg19 weights...")
    vgg_model = VGG19(include_top=False, weights="imagenet")

    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("load vgg19 layer: ", vgg_layer_name)
    last_epoch = 0

# prepare generator:
train_di = DataIterator(FLAGS.data_dir + "/keypoint_train_annotations_20170909.json",
                        FLAGS.data_dir + "/keypoint_train_images_20170902/{}.jpg",
                        batch_size=batch_size,
                        data_shape=data_shape)

# setup lr multipliers for conv layers
lr_mult = dict()
for layer in model.layers:
    if isinstance(layer, Conv2D):
        #stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            biase_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[biase_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            biase_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[biase_name] = 8

        # vgg
        else:
            kernel_name = layer.weights[0].name
            biase_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[biase_name] = 2

# configure loss functions
losses = {}
losses["Mconv5_stage1_L1"] = "mean_squared_error"
losses["Mconv5_stage1_L2"] = "mean_squared_error"
losses["Mconv7_stage2_L1"] = "mean_squared_error"
losses["Mconv7_stage2_L2"] = "mean_squared_error"
losses["Mconv7_stage3_L1"] = "mean_squared_error"
losses["Mconv7_stage3_L2"] = "mean_squared_error"
losses["Mconv7_stage4_L1"] = "mean_squared_error"
losses["Mconv7_stage4_L2"] = "mean_squared_error"
losses["Mconv7_stage5_L1"] = "mean_squared_error"
losses["Mconv7_stage5_L2"] = "mean_squared_error"
losses["Mconv7_stage6_L1"] = "mean_squared_error"
losses["Mconv7_stage6_L2"] = "mean_squared_error"


loss_weights = {}
loss_weights["Mconv5_stage1_L1"] = 1
loss_weights["Mconv5_stage1_L2"] = 1
loss_weights["Mconv7_stage2_L1"] = 1
loss_weights["Mconv7_stage2_L2"] = 1
loss_weights["Mconv7_stage3_L1"] = 1
loss_weights["Mconv7_stage3_L2"] = 1
loss_weights["Mconv7_stage4_L1"] = 1
loss_weights["Mconv7_stage4_L2"] = 1
loss_weights["Mconv7_stage5_L1"] = 1
loss_weights["Mconv7_stage5_L2"] = 1
loss_weights["Mconv7_stage6_L1"] = 1
loss_weights["Mconv7_stage6_L2"] = 1

# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_di.N // batch_size
def step_decay(epoch):
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate

# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=2)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)

callbacks_list = [lrate, checkpoint, csv_logger, tb]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

# start training
model.compile(loss=losses, loss_weights=loss_weights, optimizer=multisgd, metrics=["accuracy"])

model.fit_generator(train_di,
                    steps_per_epoch=train_di.N // batch_size,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    #use_multiprocessing=True,
                    initial_epoch=last_epoch
                    )

