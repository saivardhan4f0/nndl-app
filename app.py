import numpy as np
from flask import Flask, render_template, request
import pickle
from keras.models import load_model

app = Flask(__name__)
model = load_model('CropPrediction.h5')

@app.route('/')
def first():
    return render_template('index2.html')

@app.route('/predict', methods=['post'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(np.asarray(int_features).astype(np.float32).reshape(1, 7))

    prediction = prediction[0]
    label = ['rice', 'banana', 'black gram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
             'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pegionpeas',
             'promagranate', 'rice', 'watermelon']
    thresholded = (prediction > 0.5) * 1
    ind = np.argmax(thresholded)
    output=label[ind]

    return render_template('index2.html', pred='the predicted crop is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

