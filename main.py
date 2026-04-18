import cv2
import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model("hair_disease_model.h5")

# Load labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = dict((v, k) for k, v in class_indices.items())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    label = labels[class_index]
    confidence = round(np.max(prediction) * 100, 2)

    cv2.putText(frame, f"{label} ({confidence}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Hair Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()