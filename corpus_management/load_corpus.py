import numpy as np
import torch
import os

def load_encoded_corpus(for_train=False):
    """
    A function to load the encoded corpus from a memmap file.
    
    Returns:
        Tuple[corpus_embeddings, index_present]
    """
    # Load memmap file containing embeddings
    print("Loading memmap file containing embeddings")
    memmap_path = os.path.join("data/embeddings", "corpus.memmap")
    if not os.path.exists(memmap_path):
        return None, False
        
    # Load metadata to get shape information
    meta_path = os.path.join("data/embeddings", "corpus_meta")
    with open(meta_path, 'r') as f:
        meta = eval(f.read())
        
    # Create memmap array with correct shape
    corpus_embeddings = np.memmap(memmap_path, 
                                dtype=np.float32,
                                mode='r',
                                shape=(meta['total_number'], meta['embedding_size']))
                                
    # If for train, we don't need to convert to torch tensor
    if not for_train:
        # Convert numpy memmap to torch tensor since retrieve() expects tensor
        corpus_embeddings = torch.from_numpy(corpus_embeddings).cuda() if torch.cuda.is_available() else torch.from_numpy(corpus_embeddings).to('cpu')
                                
    return corpus_embeddings, True