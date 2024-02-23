
from os import path



base_path = "./config"
dataset_path = (base_path + "/fer2013.csv")
train_path = (base_path + "/hdf5/train.hdf5")
val_path = (base_path + "/hdf5/val.hdf5")
test_path = (base_path + "/hdf5/test.hdf5")

numbers_of_emotions = 6
batch_size = 128
output_path = path.join(base_path, "output")