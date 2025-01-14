from dexter.retriever.dense.HfRetriever import HfRetriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
import json
from dexter.data.datastructures.evidence import Evidence
from typing import List
import numpy as np
import os
from configparser import ConfigParser

def encode_corpus(output_dir: str, config: ConfigParser, sample_size: int = None):
    """
    Encode corpus and save embeddings in a format compatible with dataset.py
    
    Args:
        output_dir: Directory to save the encoded corpus files
        config: ConfigParser object containing model paths
        sample_size: Number of samples to encode, if None, encode all corpus
    """
    # Initialize RetrieverDataset to load the corpus
    dataset = RetrieverDataset(
        dataset="wikimultihopqa",
        passage_dataset="wiki-musiqueqa-corpus", 
        config_path="config.ini",
        split=Split.DEV,
        tokenizer=None  # We don't need tokenization for encoding
    )
    
    # Get corpus from dataset
    _, _, corpus = dataset.qrels()
    if sample_size:
        corpus = corpus[:sample_size]
    print(f"Loaded corpus with {len(corpus)} documents")
    
    # Initialize HfRetriever with config
    retriever = HfRetriever(config=DenseHyperParams(
        query_encoder_path=config["Query-Encoder"].get("query_encoder_path"),
        document_encoder_path=config["Document-Encoder"].get("document_encoder_path")
    ))
    
    # Encode the corpus using HfRetriever
    corpus_embeddings = retriever.encode_corpus(corpus)
    
    # Move tensor to CPU before converting to numpy
    corpus_embeddings = corpus_embeddings.cpu()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as memmap file
    embedding_size = corpus_embeddings.shape[1]
    memmap_path = os.path.join(output_dir, "corpus.memmap")
    fp = np.memmap(memmap_path, dtype=np.float32, mode='w+', 
                   shape=(len(corpus), embedding_size))
    fp[:] = corpus_embeddings[:]
    fp.flush()
    
    # Save metadata
    meta = {
        'total_number': len(corpus),
        'embedding_size': embedding_size,
        'type': str(np.float32)
    }
    with open(os.path.join(output_dir, "corpus_meta"), 'w') as f:
        json.dump(meta, f)
    
    # Save lengths (using constant max length since these are embeddings)
    lengths = np.array([embedding_size] * len(corpus))
    np.save(os.path.join(output_dir, "corpus_length.npy"), lengths)
    
    print(f"Saved encoded corpus to {output_dir}")
    print(f"Generated corpus embeddings with shape: {corpus_embeddings.shape}")
    
    return corpus_embeddings

if __name__ == "__main__":
    output_dir = "data/embeddings"
    config = ConfigParser()    
    config.read("config.ini")
    encoded_corpus = encode_corpus(output_dir, config)
    print(f"Generated corpus embeddings with shape: {encoded_corpus.shape}")
