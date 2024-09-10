# %%
import dask.dataframe as dd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import StringIO, BytesIO
import logging
import json
import faiss
import random
import pandas as pd
import ast


def download_file_to_dataframe(bucket, object_name, profile_name="DataScientist"):
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


# df_ingredients = pd.read_csv("df_ingredients_embedded.csv")[
#     ["description_embedded", "NER", "title", "ingredients"]
# ]
df_ingredients = download_file_to_dataframe(
    "bs-llm-sandbox", "keenanh/df_ingredients_embedded.parquet.gzip"
)
for drop in [6071]:
    df_ingredients = df_ingredients.drop(drop).reset_index(drop=True)

df_ingredients = dd.from_pandas(df_ingredients, npartitions=4)

df_menu_item = download_file_to_dataframe(
    "bs-llm-sandbox", "keenanh/df_menu_item_embedded.parquet.gzip"
)
df_menu_item = dd.from_pandas(df_menu_item, npartitions=4)
# df_menu_item = dd.read_csv("df_menu_item_embedded.csv")[
#     [
#         "Menu_Item_ID",
#         "Item_Description_Embedded",
#         "restaurant",
#         "Restaurant_ID",
#         "Item_Name",
#         "Item_Description",
#         "Food_Category",
#         "Serving_Size",
#     ]
# ]


# Custom function to convert the vector string to a list of floats
def parse_vector(vector_str):
    return np.array(ast.literal_eval(vector_str), dtype="float16")


# Apply the parsing function to the column with vector strings
df_ingredients["description_embedded"] = df_ingredients["description_embedded"].apply(
    parse_vector, meta=("description_embedded", "object")
)
df_menu_item["Item_Description_Embedded"] = df_menu_item[
    "Item_Description_Embedded"
].apply(parse_vector, meta=("Item_Description_Embedded", "object"))


print(df_ingredients.head()["description_embedded"])
print(df_menu_item.head()["Item_Description_Embedded"])


# Function to perform nearest neighbor search using Faiss
def nearest_neighbors_faiss(ingredients_vectors, menu_item_vectors, n_neighbors=5):
    # Convert to numpy arrays
    ingredients_vectors = np.array(ingredients_vectors, dtype="float16")
    menu_item_vectors = np.array(menu_item_vectors, dtype="float16")

    # Initialize Faiss index
    dim = ingredients_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)  # Use L2 distance, for cosine use IndexFlatIP
    index.add(menu_item_vectors)

    # Perform search
    distances, indices = index.search(ingredients_vectors, n_neighbors)

    return indices, distances


# Function to perform distributed nearest neighbor search using Dask
def distributed_nn_search(df_ingredients, df_menu_item, n_neighbors=5):
    # Compute the entire Dask DataFrame to pandas DataFrame
    df_ingredients = df_ingredients.compute()
    df_menu_item = df_menu_item.compute()

    # Extract vectors
    ingredients_vectors = df_ingredients["description_embedded"].tolist()
    menu_item_vectors = df_menu_item["Item_Description_Embedded"].tolist()

    # Perform nearest neighbor search
    indices, distances = nearest_neighbors_faiss(
        ingredients_vectors, menu_item_vectors, n_neighbors=n_neighbors
    )

    # Gather matched items
    matched_df = []
    for i, (index_set, distance_set) in enumerate(zip(indices, distances)):
        row = df_ingredients.iloc[i].to_dict()
        for index, distance in zip(index_set, distance_set):
            match = df_menu_item.iloc[index].to_dict()
            match.update(row)
            match["similarity_score"] = 1 - distance  # Assuming using L2 distance
            matched_df.append(match)

    # Convert to Dask DataFrame for further processing if needed
    return dd.from_pandas(pd.DataFrame(matched_df), npartitions=1)


# Apply distributed nearest neighbor search
result_df = distributed_nn_search(df_ingredients, df_menu_item)

# %% [markdown]
# ## Testing Similarity Metrics on Dataset


# %%
def upload_file_obj(file, bucket, object_name, profile_name="DataScientist"):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # Set Session
    boto3.setup_default_session(profile_name=profile_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file_obj(file, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


# %%
selected_df = result_df.dropna()
training_df = []
training_temp = lambda x, y: {
    "messages": [
        {
            "role": "user",
            "content": f"{'For the following menu item, interpret the menu item, menu modifiers and flair words to generate all ingredients found in its recipe in a clear, comma-separated string format.' if random.random() >.5 else 'For the following menu item, list all ingredients in a clear, comma-separated string format'}. Menu Item Name: {x}. Ingredients: ",
        },
        {"role": "assistant", "content": f"{y}"},
    ]
}

# %%
n = 5000 if len(selected_df) < 5000 else len(selected_df)
sample_size_df = selected_df.sample(n=n, random_state=77)
rows_list = sample_size_df.to_dict(orient="records")
for row_dict in rows_list:
    training_df.append(training_temp(row_dict["Item_Name"], row_dict["NER"]))

training_buf = StringIO().seek(0)
for item in training_df:
    json_line = json.dumps(item)
    training_buf.write(json_line + "\n")
upload_file_obj(training_buf, "bs-llm-sandbox", "keenanh/recipe2m_data_train.jsonl")
