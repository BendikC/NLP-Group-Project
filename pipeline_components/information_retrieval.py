from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from pipeline_components.funky_contriever import FunkyContriever
from configparser import ConfigParser
from typing import List, Optional, Dict
import numpy as np
from pipeline_components.adore_contriever import AdoreContriever

class Retriever:
    """
    Handles information retrieval (i.e., retrieving the relevant contexts to add to the prompt).
    """
    def __init__(self, config_path: str = "config.ini", corpus: Optional[List] = None, qrels: Optional[Dict] = None):
        """
        Initializes the retriever by loading the corpus or using injected corpus and qrels.

        Args:
            config_path (str): Path to the configuration file.
            corpus (Optional[List]): Injected corpus to use instead of loading from the dataset.
            qrels (Optional[Dict]): Injected qrels to use instead of loading from the dataset.
        """
        # Parse the config file
        self.config = ConfigParser()    
        self.config.read(config_path)

        if corpus is not None and qrels is not None:
            self.corpus = corpus
            self.qrels = qrels
        else:
            # Initialize dataset loader
            dataset = RetrieverDataset(
                dataset="wikimultihopqa",
                passage_dataset="wiki-musiqueqa-corpus",
                config_path=config_path,
                split=Split.DEV
            )

            # Load corpus
            self.queries, self.qrels, self.corpus = dataset.qrels()
            self.base_dataset = dataset.base_dataset

        # Convert corpus to a mapping from indices to documents for easy retrieval
        self.corpus_mapping = {doc.id(): doc.text() for doc in self.corpus}

        # Initialize retriever for caching
        self.retriever = None
    
    def retrieve_relevant_contexts_contriever(self, queries: List[Question], k: int) -> dict[str, List[str]]:
        """
        Given queries, uses off-the-shelf Contriever IR model to compute its embedding and find the documents in the 
        corpus that most closely match this computed embedding.

        Args:
            queries (List[Question]: List of input queries as Question objects.
            k (int): Amount of top-k ranked documents to return per query

        Returns:    
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the most k relevant documents in order of relevance.
        """
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must be a non-empty list of non-empty strings.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Initialize retriever
        retriever = FunkyContriever(DenseHyperParams(
            query_encoder_path=self.config["Query-Encoder"].get("query_encoder_path"),
            document_encoder_path=self.config["Document-Encoder"].get("document_encoder_path")
        ))

        # Perform retrieval for the query
        print("Retrieving relevant contexts")
        retrieval_results = retriever.retrieve(
            corpus=self.corpus,
            queries=queries,
            top_k=k,
            score_function=CosineSimilarity(),
            return_sorted=True
        )

        # Map query IDs to the top-k relevant documents
        query_to_documents = {}
        for query_id, results in retrieval_results.items():
            relevant_docs = [
                self.corpus_mapping[doc_id] for doc_id in results.keys() if doc_id in self.corpus_mapping
            ]
            query_to_documents[query_id] = relevant_docs

        return query_to_documents
    
    def retrieve_relevant_contexts_adore(self, queries: List[Question], k: int) -> dict[str, List[str]]:
        """
        Given queries, uses the ADORE fine-tuned Contriever model to compute query embeddings and find the documents in the 
        context that most closely match this computed embedding.

        Args:
            queries (List[Question]: List of input queries as Question objects.
            k (int): Amount of top-k ranked documents to return per query

        Returns:    
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the most k relevant documents in order of relevance.
        """
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must be a non-empty list of non-empty strings.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Initialize retriever
        contriever_config = DenseHyperParams(
            query_encoder_path=self.config["Query-Encoder"].get("query_encoder_path"),
            document_encoder_path=self.config["Document-Encoder"].get("document_encoder_path")
        )
        
        retriever = AdoreContriever(contriever_config, "model_checkpoints/adore/epoch-6")

        # Perform retrieval for the query
        print("Retrieving relevant contexts")
        retrieval_results = retriever.retrieve(
            corpus=self.corpus,
            queries=queries,
            top_k=k,
            score_function=CosineSimilarity(),
            return_sorted=True
        )

        # Map query IDs to the top-k relevant documents
        query_to_documents = {}
        for query_id, results in retrieval_results.items():
            relevant_docs = [
                self.corpus_mapping[doc_id] for doc_id in results.keys() if doc_id in self.corpus_mapping
            ]
            query_to_documents[query_id] = relevant_docs

        return query_to_documents
    
    def retrieve_hard_negative_contexts_contriever(self, queries: List[Question], k: int) -> dict[str, List[str]]:
        """
        Given a list of queries, find the top-k hard negative contexts. 
        Hard negatives are documents that are related and close to the query in the vector space but do not help answer the question.
        
        Using our query and imported Contriever model, sample k hard negative contexts.
        These are the highest-ranked documents that aren't in the ground truth for a query.

        Args:
            queries (List[Question]: List of input queries as Question objects.
            k (int): Amount of top-k hard-negative documents to return.

        Returns:
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the most k relevant documents in order of relevance.
        """
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must be a non-empty list of non-empty strings.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Retrieve all oracle contexts for each query (k is set to arbitrarily large number to ensure all oracle contexts are
        # retrieved)
        oracle_contexts = self.retrieve_oracle_contexts(queries, 1000)

        # Determine the maximum number of oracle contexts across all queries
        max_oracle_count = max(len(contexts) for contexts in oracle_contexts.values())

        # Retrieve (max_oracle_count + k) contriever-relevant contexts for each query
        relevant_contexts = self.retrieve_relevant_contexts_contriever(queries, max_oracle_count + k)

        # Prepare hard negatives by excluding oracle contexts from relevant contexts
        hard_negatives = {}

        for query in queries:
            query_id = str(query.id())

            if query_id not in relevant_contexts:
                hard_negatives[query_id] = []  # No relevant contexts available
                continue

            # Contriever-relevant contexts for the query
            contriever_relevant_docs = relevant_contexts[query_id]

            # Oracle contexts for the query
            oracle_docs = set(oracle_contexts.get(query_id, []))

            # Exclude oracle contexts from relevant contexts
            non_oracle_docs = [doc for doc in contriever_relevant_docs if doc not in oracle_docs]

            # Take the top-k non-oracle documents
            hard_negatives[query_id] = non_oracle_docs[:k]

        return hard_negatives


    def retrieve_oracle_contexts(self, queries: List[Question], k: int) -> dict[str, List[str]]:
        """
        Given a list of queries, find the top-k oracle contexts associated with them.

        Args:
            queries (List[Question]):  List of input queries as Question objects.
            k (int): Number of top-k ranked documents to return per question.

        Returns:
            Dict[str, List[str]]: A dictionary mapping query IDs to k oracle documents in order of relevance.
        """
        if not queries or any(not query.text() for query in queries):
            raise ValueError("Input must be a non-empty list of valid Question objects with non-empty text.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Prepare the result dictionary
        query_to_oracle_contexts = {}
        
        for query in queries:
            query_id = str(query.id())  # Ensure question ID is in string format
            
            if query_id in self.qrels:
                # Retrieve relevant document IDs from qrels
                relevant_doc_ids = list(self.qrels[query_id].keys())
                
                # Map document IDs to their text content
                relevant_docs = [
                    self.corpus_mapping[doc_id]
                    for doc_id in relevant_doc_ids
                    if doc_id in self.corpus_mapping
                ]
                
                # Return only the top-k documents (or all if fewer than k are available)
                query_to_oracle_contexts[query_id] = relevant_docs[:k]
            else:
                # If no relevant documents are found, return an empty list for the question
                query_to_oracle_contexts[query_id] = []
        
        return query_to_oracle_contexts


    def retrieve_random_contexts(self, queries: List[Question], k: int, top_k: int) -> dict[str, List[str]]:
        """
        Randomly sample k contexts that are NOT relevant. 

        Args:
            k (int): Amount of random documents to return
            queries (List[Question]: List of input queries as Question objects.
            top_k (int): Amount of relevant docs.

        Returns:
            Dict[str, List[str]]: A dictionary mapping query IDs to k irrelevant documents in order of relevance.
        """
        if k <= 0:
            raise ValueError("The number of random contexts to retrieve must be a positive integer.")

        # Ensure k does not exceed the size of the corpus
        if k > len(self.corpus_mapping):
            raise ValueError(f"The number of requested contexts exceeds the corpus size ({len(self.corpus_mapping)}).")
        
        # Retrieve relevant contexts
        relevant_contexts = self.retrieve_relevant_contexts_contriever(queries, top_k)
        
        # Retrieve oracle contexts
        oracle_contexts = self.retrieve_oracle_contexts(queries, top_k)
        
        # Create the not_relevant_mapping
        not_relevant_mapping = {}
        
        corpus_ids = set(self.corpus_mapping.keys())

        for query in queries:
            query_id = str(query.id())  # Ensure question ID is in string format

            # Get relevant and oracle contexts for the query
            relevant_docs = set(relevant_contexts.get(query_id, []))
            oracle_docs = set(oracle_contexts.get(query_id, []))
            
            # Combine relevant and oracle contexts
            excluded_docs = relevant_docs.union(oracle_docs)

            # Identify remaining document IDs that can be sampled
            remaining_doc_ids = list(corpus_ids - excluded_docs)

            if len(remaining_doc_ids) < k:
                raise ValueError(f"Not enough non-relevant documents to sample {k} for query {query_id}.")
        
            # Randomly sample k document IDs from the remaining pool
            sampled_doc_ids = np.random.choice(remaining_doc_ids, size=k, replace=False)
            sampled_docs = [self.corpus_mapping[doc_id] for doc_id in sampled_doc_ids]
            not_relevant_mapping[query_id] = sampled_docs

        return not_relevant_mapping
    