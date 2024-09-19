# %%
import pandas as pd

# %%
df_ingredients=pd.read_csv("recipes_data.csv")
df_menu_item=pd.read_csv("dohmh-menustat-historical-1.csv")

# %%
#Data Preproccess

#Adding df_ingredients column's 'title' + 'NER' to match format of df_menu_item 'Item Description'
df_ingredients['description'] = f"{df_ingredients['title']}, {df_ingredients['NER']}"

#As you see here the 'Item Description' has the item name in it
df_menu_item.head()
from langchain.embeddings import HuggingFaceEmbeddings
# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
#modelPath = "sentence-transformers/paraphrase-albert-small-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cuda'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)
df_menu_item['Item_Description_Embedded'] = df_menu_item['Item_Description'].apply(lambda x: embeddings.embed_query(x))
df_menu_item.to_csv('df_menu_item_embedded.csv',index=False)
del df_menu_item
df_ingredients['description_embedded'] = df_ingredients['description'].apply(lambda x: embeddings.embed_query(x))
df_ingredients.to_csv('df_ingredients_embedded.csv',index=False)
del df_ingredients
