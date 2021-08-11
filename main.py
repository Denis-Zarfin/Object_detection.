seed = 0
import os

os.environ['PYTHONHASHSEED'] = str(seed)
import tensorflow as tf
import numpy as np
import random

tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

import imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream, FileVideoStream
import cv2


def detector(videoframe, facedetection, maskdetection):
    """
    The function takes video frames and two detectors and returns
    coordinates and predictions whether there is a mask
    """
    (h, w) = videoframe.shape[:2]
    blobimage = cv2.dnn.blobFromImage(videoframe, 1.0, (224, 224), (104.0, 177.0, 123.0))

    facedetection.setInput(blobimage)
    ffinding = facedetection.forward()

    face_list = []
    locations = []
    predictions = []

    for i in range(0, ffinding.shape[2]):
        credence = ffinding[0, 0, i, 2]
        if credence > 0.6:
            case = ffinding[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = case.astype("int")
            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(w - 1, x_end), min(h - 1, y_end))

            image = videoframe[y_start:y_end, x_start:x_end]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            face_list.append(image)
            locations.append((x_start, y_start, x_end, y_end))

    if len(face_list) > 0:
        face_list = np.array(face_list, dtype="float32")
        predictions = maskdetection.predict(face_list, batch_size=32)
    return (locations, predictions)


path_of_pr = r"C:\Users\Denis\Documents\ObjectDetection\res10\deploy.prototxt"
path_of_w = r"C:\Users\Denis\Documents\ObjectDetection\res10\res10_300x300_ssd_iter_140000.caffemodel"

facedetection = cv2.dnn.readNet(path_of_pr, path_of_w)
maskdetection = load_model("detection.model")

videostrim = VideoStream(src=0).start()

while True:
    videoframe = videostrim.read()
    videoframe = imutils.resize(videoframe, width=700)
    (locations, predictions) = detector(videoframe, facedetection, maskdetection)

    for (case, preds) in zip(locations, predictions):
        (x_start, y_start, x_end, y_end) = case
        (mask, not_mask) = preds

        tag = "MASK" if mask > not_mask else "NO MASK"
        maskcolor = (0, 255, 0) if tag == "MASK" else (0, 0, 255)
        tag = "{}: {:.2f}%".format(tag, max(mask, not_mask) * 100)
        cv2.putText(videoframe, tag, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    maskcolor, 2)
        cv2.rectangle(videoframe, (x_start, y_start), (x_end, y_end), maskcolor, 2)

    cv2.imshow("result", videoframe)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
videostrim.stop()
