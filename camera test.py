

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import serial
ser = serial.Serial(
        port="COM5",
        baudrate=9600,
        bytesize=8,
        parity='N',
        dsrdtr=True,
        timeout=10
    )

model = load_model("model_data/model.h5")
label = []
f = open("model_data/model.txt")
for line in f.readlines():
    label.append(line.strip())
f.close()

camera = cv2.VideoCapture(1)

while camera.isOpened():

    success, frame = camera.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    image = image.reshape(1, 100, 100, 3)
    image = image.astype(np.float32)
    image = image / 255

    predicts = model.predict(image)
    idx = np.argmax(predicts)
    if idx == 0:
        ser.write(bytearray("00100000000000\n", "UTF-8"))
    elif idx == 1:
        ser.write(bytearray("00000000000000\n", "UTF-8"))
    elif idx == 2:
        ser.write(bytearray("00010000000000\n", "UTF-8"))
    elif idx == 3:
        ser.write(bytearray("00001000000000\n", "UTF-8"))


    print(idx, label[idx])
    cv2.putText(frame, label[idx], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)

    if key_code in [ord('q'), 27]:
        break

camera.release()
cv2.destroyAllWindows()
ser.close()




