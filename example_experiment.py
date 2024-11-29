"""
Most simple (envisioned) experimental setup, for the first experiment.
"""

from pipeline_components.information_retrieval import retrieve_relevant_contexts_contriever
from pipeline_components.generate_answer import generate_answer_with_llm

###TODO Iterate over the first 1200 queries of the dev.json dataset

query = "some_query"
# Top 10 docs
k = 10

relevant_contexts = retrieve_relevant_contexts_contriever(query, k)

##TODO Combine the query with the relevant contexts in some way

prompt_response = generate_answer_with_llm(relevant_contexts)

##TODO Check if this response matches up with the ground truth, and keep track of accuracy