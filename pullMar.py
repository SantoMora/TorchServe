import requests

url = "http://172.17.0.2:5000/get/mar"

payload = ""
headers = {
  'bucketName': os.environ['BUCKET_NAME'],
  'fileName': f"{os.environ['MODEL_NAME']}/{os.environ['MODEL_NAME']}.mar"
}
response = requests.request("GET", url, headers=headers, data=payload)
open(f"model-store/{os.environ['MODEL_NAME']}.mar", "wb").write(response.content)