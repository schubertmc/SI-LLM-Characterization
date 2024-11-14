#!/usr/bin/env python

from flask import Flask, request, jsonify, render_template

from api import evaluateCSSRS, evaluateFeatures

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate_cssrs', methods=['POST'])
def evaluate():
    # Parse JSON data from the request
    data = request.json
    text_input = data.get("text_input", "")

    # Evaluate the input text with the function
    output = evaluateCSSRS(text_input)

    # Return the result as JSON
    return jsonify(output)

@app.route('/evaluate_features', methods=['POST'])
def evaluate_features():
    # Parse JSON data from the request
    data = request.json
    text_input = data.get("text_input", "")

    # Evaluate the input text with the function
    output = evaluateFeatures(text_input)

    # Return the result as JSON
    return jsonify(output)

if __name__ == '__main__':
    # Host on 0.0.0.0
    app.run(host='0.0.0.0', port=5001, debug=False)
    
    # app.run(debug=True)