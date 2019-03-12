from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

# imports the vgg16 model and discards the last 1000 neuron layer.
base_model = VGG16(include_top=False, weights='imagenet', )

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(25, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

# train_generator = train_datagen.flow_from_directory('.\picnic-ai-challenge\sorted_classes',
#                                                     target_size=(224, 224),
#                                                     color_mode='rgb',
#                                                     batch_size=32,
#                                                     class_mode='categorical',
#                                                     shuffle=True)


train_datagen = ImageDataGenerator(
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory('.\picnic-ai-challenge\sorted_classes',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory('.\picnic-ai-challenge\sorted_classes',
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        subset='validation')


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

model.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=200)

model.save("picnic_model.h5")
