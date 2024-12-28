from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.retriever.dense.Contriever import Contriever
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from configparser import ConfigParser
from typing import List
import numpy as np

class Retriever:
    """
    Handles information retrieval (i.e., retrieving the relevant contexts to add to the prompt).
    ##TODO Just plugging and playing existing state-of-the-art LLMs
    """

    @staticmethod
    def retrieve_relevant_contexts_contriever(queries: List[str], k: int) -> dict[str, List[str]]:
        """
        Given queries, uses off-the-shelf Contriever IR model to compute its embedding and find the documents in the 
        corpus that most closely match this computed embedding.

        Args:
            queries (List[str]: :ist of input query strings
            k (int): Amount of top-k ranked documents to return per query

        Returns:    
            Dict[str, List[str]]: A dictionary mapping query IDs to lists of the most k relevant documents in order of relevance.
        """
        if not queries or any(not query for query in queries):
            raise ValueError("Queries must be a non-empty list of non-empty strings.")
        
        if k <= 0:
            raise ValueError("The number of top-k documents must be a positive integer.")
        
        # Parse the config file
        config = ConfigParser()
        config.read("config.ini")

        # Format input queries as questions
        queries_as_questions = [Question(text=query) for query in queries]
        
        # Initialize dataset loader
        dataset = RetrieverDataset(
            dataset="wikimultihopqa",
            passage_dataset="wiki-musiqueqa-corpus",
            config_path="config.ini",
            split=Split.DEV
        )

        # Load queries and corpus
        _, _, corpus = dataset.qrels()

        # Convert corpus to a mapping from indices to documents for easy retrieval
        corpus_mapping = {doc.id: doc.text for doc in corpus}
    
        # Initialize retriever
        retriever = Contriever(DenseHyperParams(
            query_encoder_path=config["Query-Encoder"].get("query_encoder_path"),
            document_encoder_path=config["Document-Encoder"].get("document_encoder_path")
        ))

        # Perform retrieval for the query
        retrieval_results = retriever.retrieve(
            corpus=corpus,
            queries=queries_as_questions,
            top_k=k,
            score_function=CosineSimilarity(),
            return_sorted=True
        )

        # Map query IDs to the top-k relevant documents
        query_to_documents = {}
        for query_id, results in retrieval_results.items():
            relevant_docs = [
                corpus_mapping[doc_id] for doc_id in results.keys() if doc_id in corpus_mapping
            ]
            query_to_documents[query_id] = relevant_docs

        return query_to_documents
    
    @staticmethod
    def retrieve_hard_negative_contexts_contriever(query: str, k: int) -> str:
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

    @staticmethod
    def retrieve_oracle_contexts(query: str, k: int) -> str:
        """
        Given a query, find the top-k oracle contexts associated with it.

        Args:
            query (str): String containing the input query
            k (int): Amount of top-k ranked documents to return

        Returns:
            str?: The top-k most relevant documents (ground truth?)
        """
        pass

    @staticmethod
    def retrieve_random_contexts(k: int) -> str:
        """
        Randomly sample k contexts. 

        Args:
            k (int): Amount of random documents to return

        Returns:
            str?: K random documents from the dataset
        """
        pass
