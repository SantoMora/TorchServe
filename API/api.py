from flask import Flask, render_template, request, jsonify, send_file
import boto3
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Print out bucket names
def listBuckets(bucketName):
    s3 = boto3.resource("s3")   
    for bucket in s3.buckets.all():
        if bucket.name == bucketName:
            print(bucket.name)
            return True
    return False

def uploadFile(bucketName, fileName):
    s3 = boto3.client("s3")      
    if listBuckets(bucketName):
        s3.upload_file(
            Filename=f"/modelTrained/{fileName}",
            Bucket=bucketName,
            Key=fileName,
        )
        return True
    return False

def downloadFile(bucketName, fileName):
  try:
      s3 = boto3.resource("s3")   
      s3.Bucket(bucketName).download_file(fileName, fileName.split('/')[1])
      return True
  except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == "404":
          #print("The object does not exist.")
          return False

@app.route('/', methods=['GET'])
def home():
  return "Model Serving API v0.0.1", 200

@app.route('/save/mar', methods=['POST'])
def saveMarToS3():
  try:
    req = request.get_json()
    fileDownloaded = uploadFile(req['bucketName'], req['modelName'])
    if fileUploaded:
      return jsonify("Completed"), 200
    else:
      return jsonify("Failed"), 400
  except Exception as e:
    return str(e), 400

@app.route('/get/mar', methods=['GET'])
def getMarFromS3():
  try:
    fileDownloaded = downloadFile(request.headers['bucketName'], request.headers['fileName'])
    if fileDownloaded:
      return send_file(request.headers['fileName'].split('/')[1]), 200
    else:
      return jsonify("Failed"), 400
  except Exception as e:
    return str(e), 400

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port='5000',
        debug=True,
        use_reloader=False
    )
