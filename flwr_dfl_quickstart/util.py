import boto3
from botocore.exceptions import ClientError
import os
import logging
import pandas as pd
from io import BytesIO

from flwr.common import parameters_to_ndarrays
import pickle


def _save_and_upload_global_model(
    bucket: str,
    save_path: str,
    server_round: int,
    parameters,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
):
    """Saves parameters to a pickle file and uploads the file to an S3 bucket."""

    # Convert parameters to list of NumPy arrays
    ndarrays = parameters_to_ndarrays(parameters)
    data = {"global_parameters": ndarrays}

    # Create the pickle file path
    filename = save_path + f"parameters_round_{server_round}.pkl"

    # Save the pickle file locally
    with open(filename, "wb") as h:
        pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Upload the file to S3
    return upload_file(
        filename,
        bucket,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
    )


def upload_file(
    file_name,
    bucket,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
    object_name=None,
):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_JSON_to_dataframe(
    bucket,
    object_name,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
):
    """Download an JSON file from S3 and stream it into a pandas DataFrame.

    :param bucket: Bucket to download from
    :param object_name: S3 object name to download
    :return: DataFrame containing the data from the object, or None if an error occurs
    """
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    s3_client = boto3.client("s3")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=object_name)

        # Load the file directly into a pandas DataFrame
        df = pd.read_json(BytesIO(obj["Body"].read()), lines=True)
        return df
    except Exception as e:
        return None


def download_PQT_file_to_dataframe(bucket, object_name, profile_name="DataScientist"):
    """Download an Parquet file from S3 and stream it into a pandas DataFrame.

    :param bucket: Bucket to download from
    :param object_name: S3 object name to download
    :return: DataFrame containing the data from the object, or None if an error occurs
    """
    # Set Session
    boto3.setup_default_session(profile_name=profile_name)

    # Download the Parquet file object
    s3_client = boto3.client("s3")
    try:
        # Download the Parquet file into memory
        obj = s3_client.get_object(Bucket=bucket, Key=object_name)

        # Load the Parquet file directly into a pandas DataFrame
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        return df
    except Exception as e:
        logging.error(e)
        return None
