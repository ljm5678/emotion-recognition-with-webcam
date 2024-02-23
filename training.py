from config import emotion_config as config
import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from utils import CNN_module
from utils import epochcheckpoint
from utils import hdf5generater
from utils import img_to_arr
from utils import trainingmoniter
import argparse
import os
matplotlib.use('Agg')

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type = str,
    help = "path to specific model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type = int, default = 0,
    help = "epoch to restart training at")
args = vars(ap.parse_args())


train = ImageDataGenerator(rotation_range=20, zoom_range=0.15, horizontal_flip=True, rescale=1/255.0, fill_mode="nearest")
valid = ImageDataGenerator(rescale=1/255.0)
img = img_to_arr.ImageToArrayPreprocessor()

traingen = hdf5generater.HDF5DatasetGenerator(config.train_path, config.batch_size, aug=train,preprocessors=[img], classes=config.numbers_of_emotions)
validgen = hdf5generater.HDF5DatasetGenerator(config.val_path, config.batch_size, aug=valid,preprocessors=[img], classes=config.numbers_of_emotions)


if args["model"] is None:
    print("Creating model...")
    model = CNN_module.CNN_module.build(48, 48, 1, config.numbers_of_emotions)
    optimizer = Adam(learning_rate = 1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
else:
    print("Loading model...")
    model = load_model(args["model"])
    print("old lr: {}".format(K.get_value(model.optimizer.learning_rate)))
    K.set_value(model.optimizer.learning_rate, 1e-4)
    print("new lr: {}".format(K.get_value(model.optimizer.learning_rate)))

figpath = "./output/emotion.png"
jsonpath = "./output/emotion.json"
callbacks = [
    epochcheckpoint.EpochCheckpoint("./output/checkpoints", every=1, startAt=args["start_epoch"]),
    trainingmoniter.TrainingMonitor(figpath, jsonPath=jsonpath, startAt=args["start_epoch"])
]

model.fit(
    traingen.generator(),
    steps_per_epoch=traingen.numImages // config.batch_size,
    validation_data=validgen.generator(),
    validation_steps=traingen.numImages // config.batch_size,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)


traingen.close()
validgen.close()