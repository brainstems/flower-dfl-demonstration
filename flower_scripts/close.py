import boto3
from botocore.exceptions import ClientError
import os
import logging


def upload_file(file_obj, bucket,object_name, profile_name="DataScientist", ):
    """Upload a file to an S3 bucket

    :param file_obj: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name

    # Set Session
    boto3.setup_default_session(profile_name=profile_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file_obj(file_obj, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

from datetime import datetime
with open("log.txt", "rb") as f:
    upload_file(f, "bs-llm-sandbox",f"keenanh/ingred_model/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt", "DataScientist")
