import json
import pandas as pd

## import dotenv and load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["huggingface_token"] = os.getenv("huggingface_token")
os.environ["OPENAI_KEY"] = os.getenv("OPENAI_KEY")


import json
import pandas as pd

from pipeline_components.information_retrieval import Retriever
from pipeline_components.generate_answer import AnswerGenerator
from pipeline_components.scoring_metrics import ScoringMetrics
from pipeline_components.scoring_metrics_viz import ScoringMetricsVisualization

from torch import Tensor
from typing import List,Dict
from tqdm import tqdm

TOP_K = 5

if __name__=="__main__":
        question_df = {"questions":[],"answers":[], "correct_answers":[]}
        metrics_df = {}

        retriever = Retriever()
        queries = retriever.queries
        base_dataset = retriever.base_dataset

        answer_generator = AnswerGenerator(no_print=True)

        # retrieve a dictionary of top k relevant contexts per question
        top_k_per_question = retriever.retrieve_relevant_contexts_contriever(queries, TOP_K)

        current_query_id = None

        for sample in tqdm(base_dataset.raw_data, desc="Processing samples"):
                query = sample.question
                answer = sample.answer

                query_id = query.id()
                query_text = query.text()
                relevant_contexts = top_k_per_question[query_id]

                # Because dexter package is garbage, they defined a sample in the raw data per evidence and not per query, so we have 10 samples per query
                # containing not only new evidence for each one, but the same question and answer 10 times :(
                if current_query_id == query_id:
                        continue
                else:
                        current_query_id = query_id

                prompt_context = "".join(relevant_contexts)
                final_answer = answer_generator.generate_answer_with_llm(len(prompt_context), query_text, prompt_context)

                question_df["questions"].append(query_text)
                question_df["answers"].append(final_answer)
                question_df["correct_answers"].append(answer.text())
                break

        
        answers = question_df["answers"]
        correct_answers = question_df["correct_answers"]
        metrics = {
            "Exact Match Score": (ScoringMetrics.exact_match, ScoringMetricsVisualization.plot_exact_match),
            "Cover Exact Match Score": (ScoringMetrics.cover_exact_match, ScoringMetricsVisualization.plot_cover_exact_match),
            "F1 Score": (ScoringMetrics.f1_score, ScoringMetricsVisualization.plot_f1_score),
            "BLEU Score": (ScoringMetrics.bleu, ScoringMetricsVisualization.plot_bleu),
            "METEOR Score": (ScoringMetrics.meteor, ScoringMetricsVisualization.plot_meteor),
        }

        for metric_name, metric_functions in metrics.items():
                metric_function, metric_viz = metric_functions
                score = metric_function(answers, correct_answers)
                metrics_df[f'{metric_name}'] = [score]
                print(f"{metric_name}: {score}")

        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("final_questions.csv")

        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv("metrics.csv")
        print("Done!")
                

        