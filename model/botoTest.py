import boto3  # pip install boto3

# Print out bucket names
def listBuckets(bucketName):
    s3 = boto3.resource("s3")   
    for bucket in s3.buckets.all():
        if bucket.name == bucketName:
            print(bucket.name)
            return True
    return False

def uploadFile(bucketName):
    s3 = boto3.client("s3")      
    if listBuckets(bucketName):
        s3.upload_file(
            Filename="hola.txt",
            Bucket=bucketName,
            Key="hola.txt",
        )

if __name__ == '__main__':
    bucketName = 'testbucket0943nfds'
    uploadFile(bucketName)