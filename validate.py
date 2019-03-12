from keras.models import load_model
from PIL import ImageFile
import numpy as np
from keras.utils import np_utils, to_categorical
from keras_applications.densenet import preprocess_input
from keras_preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True
model = load_model('picnic_model.h5')

from PIL import Image
import os, os.path

imgs = []
path = "./test/"
valid_images = [".jpg", ".gif", ".jpeg", ".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append((Image.open(os.path.join(path, f)), f.title()))
imgra = []

for i, name in imgs:
    i.load()

    imgr = np.resize(i, (1, 224, 224, 3))
    imgra.append((imgr, name))

del imgs

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('.\picnic-ai-challenge\sorted_classes',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
label_map = train_generator.class_indices

swapped = dict((v,k) for k,v in label_map.items())
print(swapped)

with open("result3.tsv","w+") as file :
    for img, name in imgra:
        max = model.predict(img)
        max = max.argmax()
        file.write("%s\t%s\n" % (name, swapped[max]))

