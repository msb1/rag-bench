


def list_files_in_s3_folder(bucket: str, prefix: str, client):
    """List all files in the specified S3 folder."""

    # Create a reusable paginator
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    # Iterate through every page of results
    keys = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
    return keys