import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# ------------  Load Data  -------------

name_list = pd.read_csv('YOUR DATABASE HERE.csv')
name_list = name_list[['gvkey','year','cusip','exec_fullname','execid','title']].drop_duplicates().dropna(subset=['exec_fullname'])

# -----------  Encode Name  ------------

unique_name = name_list[['execid','exec_fullname']].drop_duplicates().reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create a dataset
dataset = TextDataset(unique_name['exec_fullname'].tolist())

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Encode name in batches
embeddings = []
for i, batch in enumerate(dataloader):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, max_length=128 ,truncation=True)
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings.extend(outputs.last_hidden_state[:, 0, :].detach().to("cpu").tolist())  # Take the [CLS] token embedding
    print(f'Batch ID: {i+1} / {len(dataloader) - (i+1)}')
  
# Name Embedding
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['execid'] = unique_name['execid'].tolist()
embeddings_df['exec_fullname'] = unique_name['exec_fullname'].tolist()


# -----------  Retrieve Matched Name  ------------

def find_top_matching_execs(year, cusip, query_string, df_embeddings, df_exec,top_k=5, threshold=0.8):
    """
    Finds the top-k matching executives based on cosine similarity of name embedding.

    Args:
        year (int): The target year.
        cusip (str): The target cusip.
        query_string (str): The query to encode.
        df_embeddings (pd.DataFrame): DataFrame where first 384 cols are exec embeddings, last two are execid and exec_fullname.
        df_exec (pd.DataFrame): DataFrame with columns ['year', 'cusip', 'execid'].
        top_k (int, optional): Number of top matches to return. Defaults to 5.
        threshold (float, optional): Minimum similarity score to consider a match. Defaults to 0.8.

    Returns:
        List[Tuple[str, float, float]]: List of tuples containing exec_fullname execid and similarity score.
    """
    
    # Step 1: Get unique execids for the given year and cusip
    exec_ids = df_exec[(df_exec["year"] == year) & (df_exec["cusip"] == cusip)]["execid"].unique()
    
    # Step 2: Filter embeddings dataframe to only include these execids
    filtered_df = df_embeddings[df_embeddings["execid"].isin(exec_ids)]
    
    if filtered_df.empty:
        return []

    # Step 3: Encode the query string
    inputs = tokenizer(query_string, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    query_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # (1, 384)

    # Step 4: Compute cosine similarity
    exec_embeddings = filtered_df.iloc[:, :384].values  # Extract only the embedding columns
    similarities = cosine_similarity(query_embedding, exec_embeddings)[0]  # Compute similarity
    
    # Step 5: Filter results above threshold and get top-k matches
    filtered_results = [
        (filtered_df.iloc[i]["exec_fullname"], filtered_df.iloc[i]["execid"], similarities[i]) 
        for i in range(len(similarities)) if similarities[i] >= threshold
    ]

    # Sort by similarity and return top-k
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    return filtered_results[:top_k]

find_top_matching_execs(2004,'00790310','Hector Ruiz',embeddings_df, name_list)
