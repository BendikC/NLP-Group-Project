from dexter.retriever.dense.HfRetriever import HfRetriever
from dexter.data.datastructures.evidence import Evidence
from typing import List
import numpy as np
import os
import torch

class FunkyContriever(HfRetriever):
    def __init__(self, config=None):
        super().__init__(config)

    def load_index_if_available(self):
        """
        Override of HfRetriever's load_index_if_available to load pre-encoded corpus from memmap.
        This method will be called by retrieve() since we inherit from HfRetriever.
        
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
                                    
        # Convert numpy memmap to torch tensor since retrieve() expects tensor
        corpus_embeddings = torch.from_numpy(corpus_embeddings).cuda()
                                    
        return corpus_embeddings, True
