from experiment import Experiment, ExperimentType

if __name__ == "__main__":
    experiment = Experiment(experiment_type=ExperimentType.RELEVANT_CONTEXTS, TOP_K=5)
    experiment.run_experiment(test_mode=True)
    # experiment.run_metrics_on_saved_results("results/oracle_contexts_top_k_10-4o-mini/final_questions.csv")
                

        