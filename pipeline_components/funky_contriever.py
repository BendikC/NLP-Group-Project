from dexter.retriever.dense.HfRetriever import HfRetriever
from dexter.data.datastructures.evidence import Evidence
from typing import List
import numpy as np
import os
import torch
from corpus_management.encode_corpus import encode_corpus
from dexter.data.datastructures.question import Question
from dexter.utils.metrics.SimilarityMatch import SimilarityMetric
from typing import Dict, List
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
        corpus_embeddings = torch.from_numpy(corpus_embeddings).cuda() if torch.cuda.is_available() else torch.from_numpy(corpus_embeddings).to('cpu')
                                    
        return corpus_embeddings, True

    def retrieve(self, 
               corpus: List[Evidence], 
               queries: List[Question], 
               top_k: int, 
               score_function: SimilarityMetric,
               return_sorted: bool = True, 
               chunk: bool = False,
               chunksize = None,
               **kwargs) -> Dict[str, Dict[str, float]]:

            
        self.logger.info("Encoding Queries...")

        query_embeddings = self.encode_queries(queries, batch_size=self.batch_size)
          
        if chunk:
            results = self.retrieve_in_chunks(corpus, 
                                              queries,top_k=top_k,
                                              score_function=score_function,return_sorted=return_sorted,
                                              chunksize=chunksize)
            return results
   
        self.logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        #self.logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        embeddings, index_present = self.load_index_if_available()

        if index_present:
            corpus_embeddings = embeddings
        else:
            self.logger.info("Index not found, encoding corpus...")
            corpus_embeddings = encode_corpus("data/embeddings", self.config)
            

        # Compute similarites using either cosine-similarity or dot product
        cos_scores = score_function.evaluate(query_embeddings,corpus_embeddings)
        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        response = {}
        for idx, q in enumerate(queries):
            response[q.id()] = {}
            for index, id in enumerate(cos_scores_top_k_idx[idx]):
                response[q.id()][corpus[id].id()] = float(cos_scores_top_k_values[idx][index])
        return response