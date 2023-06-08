# organize dataset into a useful structure
from os import listdir
from os import makedirs
from random import random
from random import seed
from shutil import copyfile

from numpy import asarray
from numpy import save
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img

IMG_SIZE = 224

makedirs(f"formatted_data_{IMG_SIZE}", exist_ok=True)

# define location of dataset
folder = "train/"
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
    # determine class
    output = 0.0
    if file.startswith("dog"):
        output = 1.0
    # load image
    photo = load_img(folder + file, target_size=(IMG_SIZE, IMG_SIZE))
    # convert to numpy array
    photo = img_to_array(photo)
    # store
    photos.append(photo)
    labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save("dogs_vs_cats_photos.npy", photos)
save("dogs_vs_cats_labels.npy", labels)

# create directories
dataset_home = f"formatted_data_{IMG_SIZE}/"
subdirs = ["train/", "test/"]
for subdir in subdirs:
    # create label subdirectories
    labeldirs = ["dogs/", "cats/"]
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = "train/"
for file in listdir(src_directory):
    src = src_directory + file
    dst_dir = "train/"
    if random() < val_ratio:
        dst_dir = "test/"
    if file.startswith("cat"):
        dst = dataset_home + dst_dir + "cats/" + file
        copyfile(src, dst)
    elif file.startswith("dog"):
        dst = dataset_home + dst_dir + "dogs/" + file
        copyfile(src, dst)
