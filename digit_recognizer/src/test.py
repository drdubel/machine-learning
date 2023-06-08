import os

import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model("models/saved_model.h5")
imgs = []

for filename in os.listdir("data"):
    img = cv2.resize(cv2.imread(os.path.join("data", filename)), (28, 28))
    img = img[:, :, 0]
    img = img.reshape(1, 28, 28)
    img = img.astype("float32")
    img /= 255
    prediction = model.predict(img)
    print(f"{filename} = {prediction.argmax()}")

    plt.gray()
    plt.imshow(img[0])
    plt.show()
