import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

model = load_model("models/catsndogs-finalmodel-224.h5")
cam = cv2.VideoCapture(0)

labels = ["Cat", "Dog"]

while True:
    ret, frame = cam.read()

    if not ret:
        break

    orig_frame = frame

    frame = np.array([cv2.resize(frame, (224, 224))])

    confidence = model.predict(frame)[0][0]
    label_idx = round(confidence)

    cv2.putText(
        orig_frame,
        labels[label_idx]
        + " "
        + str((confidence if label_idx else 1 - confidence) * 100)
        + "%",
        (100, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.imshow("Cat or Dog", orig_frame)

    print(labels[label_idx], 1 - confidence)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
