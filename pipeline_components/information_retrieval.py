from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from configparser import ConfigParser
from typing import List, Optional, Dict
import numpy as np

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

        # Convert corpus to a mapping from indices to documents for easy retrieval
        self.corpus_mapping = {doc.id: doc.text for doc in self.corpus}

    def retrieve_relevant_contexts_contriever(self, queries: List[str], k: int) -> dict[str, List[str]]:
        """
        Given queries, uses off-the-shelf Contriever IR model to compute its embedding and find the documents in the 
        corpus that most closely match this computed embedding.

        Args:
            queries (List[str]: List of input query strings
            k (int): Amount of top-k ranked documents to return per query

        Returns:    
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the most k relevant documents in order of relevance.
        """
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must be a non-empty list of non-empty strings.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Format input queries as questions
        queries_as_questions = [Question(text=query) for query in queries]
    
        # Initialize retriever
        retriever = Contriever(DenseHyperParams(
            query_encoder_path=self.config["Query-Encoder"].get("query_encoder_path"),
            document_encoder_path=self.config["Document-Encoder"].get("document_encoder_path")
        ))

        # Perform retrieval for the query
        retrieval_results = retriever.retrieve(
            corpus=self.corpus,
            queries=queries_as_questions,
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
    
    def retrieve_hard_negative_contexts_contriever(self, query: str, k: int) -> str:
        """
        Using our query and imported Contriever model, sample k hard negative contexts.
        These are the highest-ranked documents that aren't in the ground truth for a query.

        Args:
            query (str): String containing the input query
            k (int): Amount of top-k hard-negative documents to return

        Returns:
            str?: K hard negative documents from the dataset
        """
        pass

    def retrieve_oracle_contexts(self, queries: List[str], k: int) -> dict[str, List[str]]:
        """
        Given a query, find the top-k oracle contexts associated with it.

        Args:
            query (List[str]): List of input query strings
            k (int): Amount of top-k ranked documents to return per query

        Returns:
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the oracle documents in order of relevance.
        """
        pass

    def retrieve_random_contexts(self, k: int) -> str:
        """
        Randomly sample k contexts. 

        Args:
            k (int): Amount of random documents to return

        Returns:
            str?: K random documents from the dataset
        """
        pass
