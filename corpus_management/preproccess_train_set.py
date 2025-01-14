import json
import os
import numpy as np
from tqdm import tqdm
import logging
import pickle
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from configparser import ConfigParser

logger = logging.getLogger(__name__)

def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

def preprocess_train_queries(config_path="config.ini", output_dir="data/preprocessed"):
    """
    Preprocesses training queries from data/qa/train.json using the query encoder tokenizer.
    Saves tokenized queries and metadata to output_dir.
    
    Args:
        config_path (str): Path to config file containing model paths
        output_dir (str): Directory to save preprocessed data
    """
    logger.info("Preprocessing training queries...")
    
    # Load config
    config = ConfigParser()
    config.read(config_path)
    
    # Initialize tokenizer from query encoder
    model = Contriever(DenseHyperParams(
        query_encoder_path=config["Query-Encoder"].get("query_encoder_path"),
        document_encoder_path=config["Document-Encoder"].get("document_encoder_path")
    ))
    tokenizer = model.question_tokenizer

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training queries
    with open("data/qa/train.json", "r") as f:
        train_data = json.load(f)
    
    # Initialize arrays to store tokenized data
    max_query_length = 64  # Match preprocess.py
    total_queries = len(train_data)
    
    # Create memory mapped arrays
    token_ids_array = np.memmap(
        os.path.join(output_dir, "train-query.memmap"),
        shape=(total_queries, max_query_length), 
        mode='w+', 
        dtype=np.int32
    )
    
    token_length_array = []
    qid2offset = {}
    
    # Process each query
    for idx, item in enumerate(tqdm(train_data, desc="Tokenizing queries")):
        query = item["question"]
        
        # Tokenize like preprocess.py
        tokens = tokenizer.encode(
            query,
            add_special_tokens=True,
            max_length=max_query_length,
            truncation=True
        )
        
        # Get length and pad
        length = min(len(tokens), max_query_length)
        input_ids = pad_input_ids(tokens, max_query_length)
        
        # Store token ids and length
        token_ids_array[idx] = input_ids
        token_length_array.append(length)
        qid2offset[idx] = idx

    # Save lengths
    np.save(os.path.join(output_dir, "train-query_length.npy"), np.array(token_length_array))
    
    # Save qid2offset mapping
    with open(os.path.join(output_dir, "train-qid2offset.pickle"), 'wb') as f:
        pickle.dump(qid2offset, f, protocol=4)
        
    # Save qrels
    with open(os.path.join(output_dir, "train-qrel.tsv"), "w") as f:
        for idx, item in enumerate(train_data):
            # Each context entry is treated as a positive context
            for context_idx, _ in enumerate(item["context"]):
                f.write(f"{idx}\t0\t{context_idx}\t1\n")

    # Save metadata
    meta = {
        "total_number": total_queries,
        "embedding_size": max_query_length,
        "type": "int32"
    }
    
    with open(os.path.join(output_dir, "train-query_meta"), "w") as f:
        json.dump(meta, f)

    logger.info(f"Preprocessed {total_queries} queries and saved to {output_dir}")

if __name__ == "__main__":
    preprocess_train_queries()
