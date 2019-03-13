# Preprocessing steps:
import os
import shutil
import pandas as pd

sorted_folder = "./sorted_classes"
train_folder = "./sorted_classes/train"
test_folder = "./sorted_classes/test"
from_folder = "./train/The Picnic Hackathon 2019/train/"

train_metadata = pd.read_csv('train.tsv', sep='\t', header=0)

train_split = 0.8

total_train = 0
total_test = 0

# sort images by label
for row, column in train_metadata.iterrows():
    if not os.path.exists('sorted_classes'):
        os.makedirs('sorted_classes')
    if not os.path.exists('sorted_classes/' + column.label):
        os.makedirs('sorted_classes/' + column.label)
    # Change to match proper directory of data
    data_from = from_folder + column.file
    data_to = './sorted_classes/' + column.label + '/' + column.file

    shutil.copyfile(data_from, data_to)

def createFolder(folder_name, image_class):
    if not os.path.exists('sorted_classes/' + folder_name):
        os.makedirs('sorted_classes/%s' % folder_name)
    if not os.path.exists('sorted_classes/%s/%s' % (folder_name, image_class)):
        os.makedirs('sorted_classes/%s/%s' % (folder_name, image_class))

    print("Created folder for class [%s] in folder [%s]" % (image_class, folder_name))
    return 'sorted_classes/%s/%s' % (folder_name, image_class)

def copyFiles(files_array, from_dir, to_dir):
    try:
        for file in files_array:
            shutil.copy(getPathForClass(from_dir, file), getPathForClass(to_dir, file))
    except Exception as e:
        print("error: %s" % e)

def getPathForClass(folder, class_name):
    return "%s/%s" % (folder, class_name)

for dirname, _, files in os.walk(sorted_folder, followlinks=False):
    if len(files) == 0:
        print("Skipping, folder empty.")
        continue

    image_class = dirname.split("\\")[-1]
    print("Found class [%s]" % image_class)

    train_folder = createFolder("train", image_class)
    test_folder = createFolder("test", image_class)

    total_images = len(files)
    train_index = int(round(total_images * train_split))

    print("Received [%s] as train folder, [%s] as test folder" % (train_folder, test_folder))
    print("Copying [%d] files for train and [%d] files for test" % (len(files[:train_index]), len(files[train_index + 1:])))
    copyFiles(files[:train_index], dirname, train_folder)
    copyFiles(files[train_index + 1:], dirname, test_folder)

    print("")
    total_train += train_index
    total_test += len(files[train_index + 1:])

print("I copied [%d] files to train on, and [%d] files to test on." % (total_train, total_test))

