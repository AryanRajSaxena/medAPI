import pandas as pd
from flask import Flask, request, jsonify
import pickle
import numpy as np

model =pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods = ['POST'])
def predict():
    text = request.form.get('text')

    input_query = text
    with open("sample.txt", "w") as f:
        f.write(text)

    dataframe = pd.read_csv(r"sample.txt")
    x_tt = dataframe.iloc[:, :]
    x_tt_transpose = np.transpose(x_tt)
    result = model.predict(x_tt_transpose)
    return jsonify({'disease':result})


if __name__ == '__main__':
    app.run(debug=True)
