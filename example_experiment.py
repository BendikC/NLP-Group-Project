from pipeline_components.information_retrieval import retrieve_relevant_contexts_contriever
from pipeline_components.generate_answer import generate_answer_with_llm

import json

# Load the dataset
with open("data/dev.json", "r") as f:
    dataset = json.load(f)

# Initialize variables to keep track of accuracy
correct_responses = 0
total_queries = 0

# Number of queries to iterate over
num_queries = 1200

# Iterate over the first 1200 queries
for i, query_data in enumerate(dataset[:num_queries]):
    query = query_data["question"]
    ground_truth = query_data["answer"]

    # Retrieve top-k relevant contexts
    k = 10
    relevant_contexts = retrieve_relevant_contexts_contriever(query, k)

    ## TODO Prompt engineering based off of the dexter prompts
    # Combine query and relevant contexts into a prompt for the LLM
    prompt = f"Question: {query}\nContext: {relevant_contexts}\nAnswer:"
    generated_response = generate_answer_with_llm(prompt)

    # Compare generated response with the ground truth
    if generated_response.strip() == ground_truth.strip():
        correct_responses += 1
    
    total_queries += 1

# Calculate and print accuracy
accuracy = correct_responses / total_queries * 100
print(f"Accuracy: {accuracy:.2f}%")
