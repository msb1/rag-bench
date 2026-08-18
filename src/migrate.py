import boto3
from botocore.config import Config

# Configure connection parameters
SOURCE_CONFIG = {
    "endpoint_url": "http://localhost:9000",
    "aws_access_key_id": "msb",
    "aws_secret_access_key": "secret",
    "bucket_name": "openrag"
}

DEST_CONFIG = {
    "endpoint_url": "http://192.168.1.50:9000",
    "aws_access_key_id": "access",
    "aws_secret_access_key": "secret",
    "bucket_name": "openrag"
}

# Use standard path style naming for custom S3 endpoints like RustFS
s3_config = Config(s3={'addressing_style': 'path'})

# Initialize clients
source_client = boto3.client('s3', 
                             endpoint_url=SOURCE_CONFIG["endpoint_url"],
                             aws_access_key_id=SOURCE_CONFIG["aws_access_key_id"],
                             aws_secret_access_key=SOURCE_CONFIG["aws_secret_access_key"],
                             config=s3_config)

dest_client = boto3.client('s3', 
                           endpoint_url=DEST_CONFIG["endpoint_url"],
                           aws_access_key_id=DEST_CONFIG["aws_access_key_id"],
                           aws_secret_access_key=DEST_CONFIG["aws_secret_access_key"],
                           config=s3_config)

def migrate_bucket():
    # Ensure destination bucket exists
    try:
        dest_client.head_bucket(Bucket=DEST_CONFIG["bucket_name"])
    except dest_client.exceptions.ClientError:
        print(f"Creating destination bucket: {DEST_CONFIG['bucket_name']}")
        dest_client.create_bucket(Bucket=DEST_CONFIG["bucket_name"])

    # List and iterate through objects in the source bucket
    paginator = source_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=SOURCE_CONFIG["bucket_name"])

    for page in pages:
        if 'Contents' not in page:
            print("Source bucket is empty.")
            return

        for obj in page['Contents']:
            key = obj['Key']
            print(f"Migrating: {key} ({obj['Size']} bytes)")

            # Stream object data from source
            response = source_client.get_object(Bucket=SOURCE_CONFIG["bucket_name"], Key=key)
            
            # Upload stream directly to destination
            dest_client.put_object(
                Bucket=DEST_CONFIG["bucket_name"],
                Key=key,
                Body=response['Body'].read(),
                ContentType=response.get('ContentType', 'application/octet-stream')
            )

if __name__ == "__main__":
    migrate_bucket()
    print("Migration complete!")
