import json
import pandas as pd

import os
os.environ["huggingface_token"] = os.getenv("huggingface_token")
os.environ["OPENAI_KEY"] = os.getenv("OPENAI_KEY")

import json
import pandas as pd

from pipeline_components.information_retrieval import Retriever
from pipeline_components.generate_answer import AnswerGenerator

from torch import Tensor
from typing import List,Dict
from tqdm import tqdm

TOP_K = 5

if __name__=="__main__":
        question_df = {"questions":[],"answers":[], "correct_answers":[]}

        retriever = Retriever()
        queries = retriever.queries
        base_dataset = retriever.base_dataset

        answer_generator = AnswerGenerator()

        # retrieve a dictionary of top k relevant contexts per question
        top_k_per_question = retriever.retrieve_relevant_contexts_contriever(queries, TOP_K)

        matches = 0
        mismatches = 0

        for sample in tqdm(base_dataset.raw_data, desc="Processing samples"):
                query = sample.question
                answer = sample.answer

                query_id = query.id()
                query_text = query.text()
                relevant_contexts = top_k_per_question[query_id]

                prompt_context = "".join(relevant_contexts)
                final_answer = answer_generator.generate_answer_with_llm(len(prompt_context), query_text, prompt_context)

                question_df["questions"].append(query_text)
                question_df["answers"].append(final_answer)
                question_df["correct_answers"].append(answer.text())

                if len(final_answer.split("[Final Answer]:")) >1:
                        final_answer = final_answer.split("[Final Answer]:")[1]
                        if answer.text().strip().lower() in final_answer.strip().lower():
                                matches += 1
                        else:
                                mismatches += 1
                else:
                        mismatches += 1
                
                final_questions = pd.DataFrame(question_df)
                print("EM", matches/(matches+mismatches))
                final_questions.to_csv("final_questions.csv")
                

        