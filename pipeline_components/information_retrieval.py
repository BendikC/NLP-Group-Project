"""
This file handles the information retrieval logic, i.e retrieving the relevant 
contexts to add to the prompt. 
    ##TODO Just plugging and playing existing state of the art LLMs
"""

def retrieve_relevant_contexts_contriever(query: str, k: int) -> str:
    """
    Given a query, we should use the already implemented contriever 
    IR model to compute its embedding and find the documents in the 
    corpus that most closely match this computed embedding.

    Args:
        query (str): String containing the input query
        k (int): Amount of top-k ranked documents to return

    Returns:
        str?: The top k most relevant documents
    """
    pass

def retrieve_hard_negative_contexts_contriever(query: str, k: int) -> str:
    """
    Using our query, and our imported contriever model, we want to sample
    k hard negative contexts. We can find these by accessing the ground
    truth documents and seeing the highest ranked documents that aren't 
    in the list of ground truth documents for a query.

    Args:
        query (str): String containing the input query
        k (int): Amount of top-k nard-negative documents to return

    Returns:
        str?: K random documents from our dataset
    """
    pass

def retrieve_oracle_contexts(query: str, k: int) -> str:
    """
    Given a query, we should find the top-k oracle contexts associated with it.

    Args:
        query (str): String containing the input query
        k (int): Amount of top-k ranked documents to return

    Returns:
        str?: The top k most relevant documents (ground truth?)
    """
    pass

def retrieve_random_contexts(k: int) -> str:
    """
    Randomly sample k contexts. 

    Args:
        k (int): Amount of random documents to return

    Returns:
        str?: K random documents from our dataset
    """
    pass




