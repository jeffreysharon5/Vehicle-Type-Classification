import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from decision import decision_logic
import matplotlib.pyplot as plt

model = load_model('model/vehicle_model.h5')

classes = ['ambulance', 'bike', 'bus', 'car', 'truck']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))

    predicted_class = classes[class_index]
    decision = decision_logic(confidence)

    print("\n--- Prediction Result ---")
    print("Predicted Class :", predicted_class)
    print("Confidence      :", round(confidence, 2))
    print("Decision        :", decision)

    plt.imshow(img)
    plt.title(f"{predicted_class} ({round(confidence,2)})")
    plt.axis('off')
    plt.show()

predict_image("test.jpg")