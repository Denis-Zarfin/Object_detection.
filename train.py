seed = 0
import os

os.environ['PYTHONHASHSEED'] = str(seed)
import tensorflow as tf
import numpy as npg
import random

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

dataset_path = r"C:\Users\Denis\Documents\ObjectDetection\data"
my_classes = ["mask", "not_mask"]

dataset = []
tags = []

for my_class in my_classes:
    pathway = os.path.join(dataset_path, my_class)
    for photo in os.listdir(pathway):
        photo_pathway = os.path.join(pathway, photo)
        processed_picture = load_img(photo_pathway, target_size=(224, 224))
        processed_picture = img_to_array(processed_picture)
        processed_picture = preprocess_input(processed_picture)
        dataset.append(processed_picture)
        tags.append(my_class)

label = LabelBinarizer()
tags = label.fit_transform(tags)
tags = to_categorical(tags)

dataset = np.array(dataset, dtype="float32")
tags = np.array(tags)

(x_train, x_test, y_train, y_test) = train_test_split(dataset, tags, test_size=0.3, stratify=tags)
(x_val, x_test, y_val, y_test) = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)

augmentation = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
trailModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = trailModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.55)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=trailModel.input, outputs=headModel)

for layer in trailModel.layers:
    layer.trainable = False

lrt = 1e-4
EPOCHS = 40
batch = 32

optimizer = Adam(lr=lrt, decay=lrt / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

History = model.fit(augmentation.flow(x_train, y_train, batch_size=batch),
                    steps_per_epoch=len(x_train) // batch,
                    validation_data=(x_val, y_val),
                    validation_steps=len(x_val) // batch,
                    epochs=EPOCHS)

pred = model.predict(x_test, batch_size=batch)
pred = np.argmax(pred, axis=1)

print(classification_report(y_test.argmax(axis=1), pred, target_names=label.classes_))

y_test = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test, pred)

print(cm)

model.save("detection.model", save_format="h5")

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, EPOCHS), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), History.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), History.history["val_accuracy"], label="val_accuracy")
plt.xlabel("EPOCH")
plt.ylabel("loss_and_acc")
plt.legend(loc="lower left")
plt.savefig("plot.png")
