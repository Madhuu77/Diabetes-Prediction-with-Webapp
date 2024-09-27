import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('classifier.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'The Diabetes Result is: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)