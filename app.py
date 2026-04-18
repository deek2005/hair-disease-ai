from datetime import datetime
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import json

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("hair_disease_model.h5")

# Load labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# Remedies
remedies = {
    "Alopecia Areata": {"natural": "Onion juice, castor oil", "medical": "Corticosteroids"},
    "Contact Dermatitis": {"natural": "Avoid irritants, aloe vera", "medical": "Antihistamines"},
    "Folliculitis": {"natural": "Warm compress", "medical": "Antibiotics"},
    "Head Lice": {"natural": "Neem oil", "medical": "Permethrin lotion"},
    "Lichen Planus": {"natural": "Aloe vera", "medical": "Steroid creams"},
    "Male Pattern Baldness": {"natural": "Healthy diet", "medical": "Minoxidil"},
    "Psoriasis": {"natural": "Coconut oil", "medical": "Coal tar shampoo"},
    "Seborrheic Dermatitis": {"natural": "Tea tree oil", "medical": "Ketoconazole shampoo"},
    "Telogen Effluvium": {"natural": "Reduce stress", "medical": "Consult doctor"},
    "Tinea Capitis": {"natural": "Keep scalp clean", "medical": "Antifungal meds"}
}

# 🌍 Language translations
translations = {
    "en": {
        "natural": "Natural Treatment",
        "medical": "Medical Treatment",
        "report": "Hair Disease AI Report",
        "name": "Patient Name",
        "age": "Age"
    },
    "hi": {
        "natural": "प्राकृतिक उपचार",
        "medical": "चिकित्सीय उपचार",
        "report": "हेयर रोग एआई रिपोर्ट",
        "name": "रोगी का नाम",
        "age": "आयु"
    },
    "kn": {
        "natural": "ಸ್ವಾಭಾವಿಕ ಚಿಕಿತ್ಸೆ",
        "medical": "ವೈದ್ಯಕೀಯ ಚಿಕಿತ್ಸೆ",
        "report": "ಕೂದಲು ರೋಗ AI ವರದಿ",
        "name": "ರೋಗಿಯ ಹೆಸರು",
        "age": "ವಯಸ್ಸು"
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    natural = ""
    medical = ""
    confidence = ""
    confidence_value = 0

    # default values (IMPORTANT to avoid error)
    patient_name = ""
    age = ""
    language = "en"
    text = translations["en"]

    if request.method == "POST":
        patient_name = request.form["patient_name"]
        age = request.form["age"]
        language = request.form["language"]

        text = translations.get(language, translations["en"])

        file = request.files["image"]
        path = "temp.jpg"
        file.save(path)

        # Preprocess
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.reshape(img, (1, 224, 224, 3))

        # Prediction
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence_raw = float(np.max(prediction))

        class_name = labels[class_index]

        confidence_value = confidence_raw * 100
        confidence = f"{confidence_value:.2f}%"

        if confidence_raw < 0.6:
            result = "⚠ Unable to confidently detect. Please upload a clearer image."
        else:
            result = class_name

        if confidence_raw >= 0.6 and class_name in remedies:
            natural = remedies[class_name]["natural"]
            medical = remedies[class_name]["medical"]

    return render_template(
        "index.html",
        result=result,
        natural=natural,
        medical=medical,
        confidence=confidence,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_name=patient_name,
        age=age,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)