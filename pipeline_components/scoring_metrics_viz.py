import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pipeline_components.scoring_metrics import ScoringMetrics


class ScoringMetricsVisualization:
    """
    A class to compute and visualize various evaluation metrics for predictions
    compared to ground truth values.
    """

    @staticmethod
    def plot_exact_match(predictions: List[str], truths: List[str], save_path=None):
        """
        Plots the Exact Match (EM) score per data point as a histogram.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.
        """
        # Compute exact match scores for each data point
        scores = ScoringMetricsVisualization.__exact_match_per_point(predictions, truths)

        # Create histogram for visualization
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("Exact Match Scores")
        plt.xlabel("Exact Match Score")
        plt.ylabel("Frequency")
        plt.xticks(ticks=[i * 0.1 for i in range(11)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.show() if save_path is None else plt.savefig(save_path)

    @staticmethod
    def __exact_match_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the Exact Match (EM) score per data point.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.

        Returns:
            List[float]: List of EM scores (1.0 if exact match, else 0.0).
        """
        # Check if each prediction matches the ground truth exactly
        return [1.0 if pred == truth else 0.0 for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_cover_exact_match(predictions: List[str], truths: List[str], save_path=None):
        """
        Plots the Cover Exact Match (cover-EM) score per data point as a histogram.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.
        """
        # Compute cover-EM scores for each data point
        scores = ScoringMetricsVisualization.__cover_exact_match_per_point(predictions, truths)

        # Create histogram for visualization
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("Cover Exact Match Scores")
        plt.xlabel("Cover Exact Match Score")
        plt.ylabel("Frequency")
        plt.xticks(ticks=[i * 0.1 for i in range(11)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.show() if save_path is None else plt.savefig(save_path)

    @staticmethod
    def __cover_exact_match_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the Cover Exact Match (cover-EM) score per data point.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.

        Returns:
            List[float]: List of cover-EM scores (1.0 if truth in pred, else 0.0).
        """
        # Check if the ground truth string is contained in the prediction
        return [1.0 if truth in pred else 0.0 for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_f1_score(predictions: List[str], truths: List[str], save_path=None):
        """
        Plots the F1 score per data point as a histogram.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.
        """
        # Compute F1 scores for each data point
        scores = ScoringMetricsVisualization.__f1_score_per_point(predictions, truths)

        # Create histogram for visualization
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("F1 Scores")
        plt.xlabel("F1 Score")
        plt.ylabel("Frequency")
        plt.xticks(ticks=[i * 0.1 for i in range(11)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.show() if save_path is None else plt.savefig(save_path)

    @staticmethod
    def __f1_score_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the F1 score per data point. Uses custom token-based computation.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.

        Returns:
            List[float]: List of F1 scores for each data point.
        """
        scores = []
        for pred, truth in zip(predictions, truths):
            # Split predictions and truths into tokens
            pred_tokens = pred.split()
            truth_tokens = truth.split()

            # Calculate precision and recall
            true_positives = len(set(pred_tokens) & set(truth_tokens))
            precision = true_positives / len(pred_tokens) if pred_tokens else 0.0
            recall = true_positives / len(truth_tokens) if truth_tokens else 0.0

            # Compute F1 score
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            scores.append(f1)
        return scores

    @staticmethod
    def plot_bleu(predictions: List[str], truths: List[str], save_path=None):
        """
        Plots the BLEU score per data point as a histogram.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.
        """
        # Compute BLEU scores for each data point
        scores = ScoringMetricsVisualization.__bleu_per_point(predictions, truths)

        # Create histogram for visualization
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("BLEU Scores")
        plt.xlabel("BLEU Score")
        plt.ylabel("Frequency")
        plt.xticks(ticks=[i * 0.1 for i in range(11)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.show() if save_path is None else plt.savefig(save_path)

    @staticmethod
    def __bleu_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the BLEU score per data point using the NLTK library.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.

        Returns:
            List[float]: List of BLEU scores for each data point.
        """
        # Use smoothing to handle cases with low n-gram overlap effectively
        smoother = SmoothingFunction()

        # Compute BLEU score for each prediction-truth pair
        return [sentence_bleu([truth.split()], pred.split(), smoothing_function=smoother.method1) 
                for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_meteor(predictions: List[str], truths: List[str], save_path=None):
        """
        Plots the METEOR score per data point as a histogram.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.
        """
        # Compute METEOR scores for each data point
        scores = ScoringMetricsVisualization.__meteor_per_point(predictions, truths)

        # Create histogram for visualization
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("METEOR Scores")
        plt.xlabel("METEOR Score")
        plt.ylabel("Frequency")
        plt.xticks(ticks=[i * 0.1 for i in range(11)])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.show() if save_path is None else plt.savefig(save_path)

    @staticmethod
    def __meteor_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the METEOR score per data point using the NLTK library.

        Args:
            predictions (List[str]): List of predicted strings.
            truths (List[str]): List of ground truth strings.

        Returns:
            List[float]: List of METEOR scores for each data point.
        """
        ScoringMetrics._ensure_nltk_resources()

        # Compute METEOR score for each prediction-truth pair
        return [
            meteor_score([truth.split()], pred.split())
            for pred, truth in zip(predictions, truths)
        ]