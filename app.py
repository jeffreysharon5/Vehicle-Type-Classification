from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from decision import decision_logic

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
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

    return predicted_class, round(confidence, 2), decision

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            pred, conf, decision = predict_image(filepath)

            return render_template('index.html',
                                   prediction=pred,
                                   confidence=conf,
                                   decision=decision,
                                   image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)