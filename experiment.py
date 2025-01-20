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
from tqdm import tqdm

from pipeline_components.information_retrieval import Retriever
from pipeline_components.generate_answer import AnswerGenerator
from pipeline_components.scoring_metrics import ScoringMetrics
from pipeline_components.scoring_metrics_viz import ScoringMetricsVisualization
from enum import Enum
from typing import Dict, List

class ExperimentType(Enum):
    RELEVANT_CONTEXTS = "relevant_contexts"
    ORACLE_CONTEXTS = "oracle_contexts"
    RANDOM_CONTEXTS = "random_contexts"
    HARD_NEGATIVE_CONTEXTS = "hard_negative_contexts"
    ADORE_CONTEXTS = "adore_contexts"


class Experiment:
    def __init__(self, experiment_type=ExperimentType.RELEVANT_CONTEXTS, TOP_K=10, NOISE_RATIO=0.2):
        # Define the number of contexts to retrieve for each query
        self.TOP_K = TOP_K
        self.NOISE_RATIO = NOISE_RATIO

        # Initializing retriever and getting queries + dataset
        self.retriever = Retriever()
        self.queries = self.retriever.queries
        self.base_dataset = self.retriever.base_dataset

        # Initializing answer generator, handling LLM logic
        self.answer_generator = AnswerGenerator(no_print=True)

        # Retrieves the contexts for the experiment, based on the experiment type
        self.experiment_contexts = self.select_contexts_for_experiment(experiment_type)

    
    def select_contexts_for_experiment(self, experiment_type: ExperimentType) -> Dict[str, List[str]]:
        """
        Retrieves the contexts for the experiment, based on the experiment type.

        Args:
            experiment_type (ExperimentType): The type of experiment to run
        Returns:
            Dict[str, List[str]]: A dictionary containing the list of contexts for each query ID
        """
        if experiment_type == ExperimentType.RELEVANT_CONTEXTS:
            return self.retriever.retrieve_relevant_contexts_contriever(self.queries, self.TOP_K)
        
        elif experiment_type == ExperimentType.ORACLE_CONTEXTS:
            return self.retriever.retrieve_oracle_contexts(self.queries, self.TOP_K)
        
        elif experiment_type == ExperimentType.RANDOM_CONTEXTS:
            # We inject noise based on number of retrieved times the noise ratio
            num_of_random = int(self.TOP_K * self.NOISE_RATIO)
            random_contexts = self.retriever.retrieve_random_contexts(self.queries, num_of_random, self.TOP_K)

            relevant_contexts = self.retriever.retrieve_relevant_contexts_contriever(self.queries, self.TOP_K)
            return self.inject_noisy_contexts(relevant_contexts, random_contexts)
        
        elif experiment_type == ExperimentType.HARD_NEGATIVE_CONTEXTS:
            # We inject noise based on number of retrieved times the noise ratio
            num_of_hard_negatives = int(self.TOP_K * self.NOISE_RATIO)
            hard_negatives = self.retriever.retrieve_hard_negative_contexts_contriever(self.queries, num_of_hard_negatives)

            relevant_contexts = self.retriever.retrieve_relevant_contexts_contriever(self.queries, self.TOP_K)
            return self.inject_noisy_contexts(relevant_contexts, hard_negatives)
        
        elif experiment_type == ExperimentType.ADORE_CONTEXTS:
            print("Gathering Adore experiment contexts...")
            return self.retriever.retrieve_relevant_contexts_adore(self.queries, self.TOP_K)
        
    def inject_noisy_contexts(self, relevant_contexts: Dict[str, List[str]],
                               noise_contexts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Injects the noisy contexts into the retrieved relevant ones. This can 
        be either the random contexts or the hard negatives.

        Args:
            relevant_contexts (Dict[str, List[str]]): The relevant contexts as a dict of lists of string per query
            noise_contexts (Dict[str, List[str]]): The contexts to add
        Returns:
            Dict[str, List[str]]: A dictionary containing the list of contexts for each query ID
        """
        resulting_contexts = {}
        
        for query_id, relevant_contexts in relevant_contexts.items():
            # Get the noise contexts from the other dictionary, slicing to correct amount
            noise_to_add = noise_contexts[query_id]
            # Add the noise to the relevant contexts
            new_contexts = relevant_contexts + noise_to_add
            resulting_contexts[query_id] = new_contexts

        return resulting_contexts
         
    def run_experiment(self, test_mode=False):
        """
        Runs the experiment, generating answers for each question and evaluating the performance of the model.
        """
        # Initialize dfs to store questions, answers, and metrics
        question_df = {"questions":[],"answers":[], "correct_answers":[]}
        metrics_df = {}

        current_query_id = None

        for sample in tqdm(self.base_dataset.raw_data, desc="Processing samples"):
                query = sample.question
                answer = sample.answer

                query_id = query.id()
                query_text = query.text()
                relevant_contexts = self.experiment_contexts[query_id]

                # In dexter package, they defined a sample in the raw data per evidence and not per query, so we have 10 samples per query
                # containing not only new evidence for each one, but the same question and answer 10 times :(
                if current_query_id == query_id:
                    print("Skipping query", query_id)
                    continue
                else:
                        current_query_id = query_id

                prompt_context = "".join(relevant_contexts)
                final_answer = self.answer_generator.generate_answer_with_llm(len(prompt_context), query_text, prompt_context)

                question_df["questions"].append(query_text)
                question_df["answers"].append(final_answer)
                question_df["correct_answers"].append(answer.text())

                # Only run the experiment for one iteration if test mode is enabled
                # This way we don't spend too much money on OpenAI API
                if test_mode:
                    break
        
        answers = question_df["answers"]
        correct_answers = question_df["correct_answers"]

        # apply the metrics to the answers, storing results in metrics_df and plots in "plots" folder
        metrics_df = self.run_metrics(answers, correct_answers, metrics_df)

        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("final_questions.csv")

        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv("metrics.csv")
        print("Done!")

    def run_metrics(self, answers: List[str], correct_answers:List[str],
                     metrics_df: Dict[str, List[int]], plot_output_folder="plots"):
        """
        Runs the metrics on the answers generated by the model and the correct answers.

        Args:
            answers (List[str]): The list of answers generated by the model
            correct_answers (List[str]): The list of correct answers
            metrics_df (Dict[str, List[int]]): The dictionary to store the metrics
            plot_output_folder (str): The folder to store the plots
        Returns:
            Dict[str, List[int]]: The dictionary containing the metrics
        """
        # Defining a dict of the metrics and methods to run, to avoid code duplication
        metrics = {
            "Exact Match Score": (ScoringMetrics.exact_match, ScoringMetricsVisualization.plot_exact_match),
            "Cover Exact Match Score": (ScoringMetrics.cover_exact_match, ScoringMetricsVisualization.plot_cover_exact_match),
            "BLEU Score": (ScoringMetrics.bleu, ScoringMetricsVisualization.plot_bleu),
            "METEOR Score": (ScoringMetrics.meteor, ScoringMetricsVisualization.plot_meteor),
        }

        if not os.path.exists(plot_output_folder):
                os.makedirs(plot_output_folder)

        # Iterating over metrics to visualize and store them
        for metric_name, metric_functions in metrics.items():
                metric_function, metric_viz = metric_functions
                score = metric_function(answers, correct_answers)
                metric_viz(answers, correct_answers, save_path=os.path.join(plot_output_folder, f"{metric_name}.png"))
                metrics_df[f'{metric_name}'] = [score]
                print(f"{metric_name}: {score}")

        return metrics_df
         
    def run_metrics_on_saved_results(self, results_path=""):
        """
        Runs the metrics on the answers generated by the model and the correct answers.

        Args:
            results_path (str): The path to the results file
        """
        results_df = pd.read_csv(results_path)
        print("Loaded results from file")
        answers = results_df["answers"].tolist()
        #remove dot at the end of the answers
        answers = [answer[:-1] if answer.endswith(".") else answer for answer in answers]
        correct_answers = results_df["correct_answers"].tolist()

        print("Running metrics on the loaded results...")
        metrics_df = {}
        metrics_df = self.run_metrics(answers, correct_answers, metrics_df)

        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv("metrics.csv")
        print("Done!")





    