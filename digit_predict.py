import cv2 as cv
import numpy as np
import tensorflow.keras as tf
import tensorflow
import os

mod =tf.models.load_model("digit.model2")


while True:

    os.system("cls")

    img =cv.imread("5.png")[:,:,0]
    arr_img = np.invert(np.array([img]))
    tf.utils.normalize(arr_img, axis=1)
    pre=mod.predict(arr_img)
    
    print(np.argmax(pre))
    input()
    

