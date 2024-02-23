import config.emotion_config as config
from utils.hdf5converter import hdf5converter
import numpy as np  # Corrected import

f = open(config.dataset_path)
next(f)  # Skip the header or first line if necessary
(train_img, train_label, val_img, val_label, test_img, test_label) = ([], [], [], [], [], [])

for rows in f:
    (label, img, usage) = rows.strip().split(",")
    label = int(label)

    if label != 0:
        label -= 1

    # Correcting the process to convert img to a numpy array and then reshape
    image = np.array(img.split(" "), dtype="uint8").reshape(48, 48)

    if usage == "Training":
        train_img.append(image)
        train_label.append(label)
    elif usage == "PrivateTest":
        val_img.append(image)
        val_label.append(label)
    else:
        test_img.append(image)
        test_label.append(label)


dataset = [(train_img, train_label, config.train_path), (val_img, val_label, config.val_path), (test_img, test_label, config.test_path)]
for (images, labels, path) in dataset:
    writer = hdf5converter((len(images), 48, 48), path)  # Assuming hdf5converter is a module with a class of the same name
    for (image, label) in zip(images, labels):
        writer.add([image], [label])
    writer.close()

f.close()
