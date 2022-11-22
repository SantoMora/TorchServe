from flask import Flask, render_template, request, jsonify
import boto3
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
  return "Model Serving API v0.0.1", 200

@app.route('/save/mar', methods=['POST'])
def saveMarToS3():
  try:
    req = request.get_json()
    alert = Outliers(req['params'], req['profile_name'], req['data'])
    if alert['completed']:
      return jsonify(alert['msn']), 200
    else:
      return jsonify(alert['msn']), 400
  except Exception as e:
    return str(e), 400

