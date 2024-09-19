# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import boto3
from botocore.exceptions import ClientError
from io import StringIO
import logging
import json
import random


df_ingredients = pd.read_csv("df_ingredients_embedded.csv")[
    ["description_embedded", "NER", "title", "ingredients"]
]


df_menu_item = pd.read_csv("df_menu_item_embedded.csv")[
    [
        "Menu_Item_ID",
        "Item_Description_Embedded",
        "restaurant",
        "Restaurant_ID",
        "Item_Name",
        "Item_Description",
        "Food_Category",
        "Serving_Size",
    ]
]
df_ingredients.to_parquet(
    "df_ingredients.parquet.gzip", engine="pyarrow", compression="gzip"
)
df_menu_item.to_parquet(
    "df_menu_item.parquet.gzip", engine="pyarrow", compression="gzip"
)
df_menu_item.head()["pb"]
# df_ingredients = df_ingredients[['description_embedded','NER','title','ingredients']]


# df_menu_item = df_menu_item[['Menu_Item_ID','Item_Description_Embedded','restaurant','Restaurant_ID','Item_Name','Item_Description','Food_Category','Serving_Size']]

# %%


def fuzzy_left_join(
    df1, df2, vector_col1, vector_col2, threshold=0.8, similarity_metric="cosine"
):
    """
    Perform a fuzzy left join on two pandas dataframes based on vector embedding similarity scores.

    Parameters:
    df1 (pd.DataFrame): Left DataFrame.
    df2 (pd.DataFrame): Right DataFrame.
    vector_col1 (str): Column name in df1 containing the vectors.
    vector_col2 (str): Column name in df2 containing the vectors.
    threshold (float): Minimum similarity score to consider a match.
    similarity_metric (str): Similarity metric to use ('cosine', 'euclidean', 'manhattan', etc.).

    Returns:
    pd.DataFrame: Resulting DataFrame after fuzzy left join.
    """
    # Extract vectors
    vectors1 = np.vstack(df1[vector_col1].values)
    vectors2 = np.vstack(df2[vector_col2].values)

    # Compute similarity matrix
    if similarity_metric == "cosine":
        similarity_matrix = cosine_similarity(vectors1, vectors2)
    else:
        distance_matrix = cdist(vectors1, vectors2, metric=similarity_metric)
        similarity_matrix = 1 - distance_matrix / np.max(distance_matrix)

    # Create an empty DataFrame to store the results
    results = []

    for i, row in df1.iterrows():
        similarities = similarity_matrix[i]
        max_similarity_index = np.argmax(similarities)
        max_similarity_score = similarities[max_similarity_index]

        if max_similarity_score >= threshold:
            match = df2.iloc[max_similarity_index].to_dict()
            match.update(row.to_dict())
            match["similarity_score"] = max_similarity_score
            results.append(match)
        else:
            row_dict = row.to_dict()
            row_dict.update({col: None for col in df2.columns})
            row_dict["similarity_score"] = None
            results.append(row_dict)

    return pd.DataFrame(results)


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
cosine_df = fuzzy_left_join(
    df_menu_item,
    df_ingredients,
    "Item_Description_Embedded",
    "description_embedded",
    threshold=0.85,
    similarity_metric="cosine",
)
cosine_df.sample().to_csv("cosine_df_sample.csv", index=False)
cos_buf = StringIO()
cosine_df.sample().to_csv(cos_buf)
upload_file_obj(cos_buf, "bs-llm-sandbox", "keenanh/cosine_df.csv")

# %%
seuclidian_df = fuzzy_left_join(
    df_menu_item[
        [
            "Menu_Item_ID",
            "Item_Description_Embedded",
            "restaurant",
            "Restaurant_ID",
            "Item_Name",
            "Item_Description",
            "Food_Category",
            "Serving_Size",
        ]
    ],
    df_ingredients[["description_embedded", "NER", "title", "ingredients"]],
    "Item_Description_Embedded",
    "description_embedded",
    threshold=0.85,
    similarity_metric="seuclidean",
)

seuc_buf = StringIO()
seuclidian_df.sample().to_csv(seuc_buf)
upload_file_obj(seuc_buf, "bs-llm-sandbox", "keenanh/seuclidian_df.csv")

# %%
jaccard_df = fuzzy_left_join(
    df_menu_item[
        [
            "Menu_Item_ID",
            "Item_Description_Embedded",
            "restaurant",
            "Restaurant_ID",
            "Item_Name",
            "Item_Description",
            "Food_Category",
            "Serving_Size",
        ]
    ],
    df_ingredients[["description_embedded", "NER", "title", "ingredients"]],
    "Item_Description_Embedded",
    "description_embedded",
    threshold=0.85,
    similarity_metric="jaccard",
)
jaccard_df.sample().to_csv("jaccard_df_sample.csv", index=False)
jaccard_buf = StringIO()
jaccard_df.sample().to_csv(jaccard_buf)
upload_file_obj(jaccard_buf, "bs-llm-sandbox", "keenanh/jaccard_df.csv")

# %%
selected_df = cosine_df.dropna()
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
for i in range(len(selected_df)):
    row_dict = selected_df.iloc[i].to_dict()
    training_df.append(training_temp(row_dict["Item_Name"], row_dict["NER"]))

training_buf = StringIO().seek(0)
for item in training_df:
    json_line = json.dumps(item)
    training_buf.write(json_line + "\n")
upload_file_obj(training_buf, "bs-llm-sandbox", "keenanh/recipe2m_data_train.jsonl")
