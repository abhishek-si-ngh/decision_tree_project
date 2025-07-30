from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/model.pkl')

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
