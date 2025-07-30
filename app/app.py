import os
import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load model using absolute path
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        features = [float(request.form.get(f)) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([features])[0]
        class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        return render_template('index.html', prediction=class_map[prediction])
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
