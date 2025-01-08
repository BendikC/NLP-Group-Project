import matplotlib.pyplot as plt
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score

class ScoringMetricsVisualization:
    @staticmethod
    def plot_exact_match(predictions: List[str], truths: List[str]):
        """
        Plots the Exact Match (EM) score per datapoint.
        """
        scores = ScoringMetricsVisualization.__exact_match_per_point(predictions, truths)
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("Exact Match Scores")
        plt.xlabel("Exact Match Score")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def __exact_match_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the Exact Match (EM) score per datapoint.
        """
        return [1.0 if pred == truth else 0.0 for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_cover_exact_match(predictions: List[str], truths: List[str]):
        """
        Plots the Cover Exact Match (cover-EM) score per datapoint.
        """
        scores = ScoringMetricsVisualization.__cover_exact_match_per_point(predictions, truths)
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("Cover Exact Match Scores")
        plt.xlabel("Cover Exact Match Score")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def __cover_exact_match_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the Cover Exact Match (cover-EM) score per datapoint.
        """
        return [1.0 if truth in pred else 0.0 for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_f1_score(predictions: List[str], truths: List[str]):
        """
        Plots the F1 score per datapoint.
        """
        scores = ScoringMetricsVisualization.__f1_score_per_point(predictions, truths)
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("F1 Scores")
        plt.xlabel("F1 Score")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def __f1_score_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the F1 score per datapoint.
        """
        scores = []
        for pred, truth in zip(predictions, truths):
            pred_tokens = pred.split()
            truth_tokens = truth.split()
            true_positives = len(set(pred_tokens) & set(truth_tokens))
            precision = true_positives / len(pred_tokens) if pred_tokens else 0.0
            recall = true_positives / len(truth_tokens) if truth_tokens else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            scores.append(f1)
        return scores

    @staticmethod
    def plot_bleu(predictions: List[str], truths: List[str]):
        """
        Plots the BLEU score per datapoint.
        """
        scores = ScoringMetricsVisualization.__bleu_per_point(predictions, truths)
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("BLEU Scores")
        plt.xlabel("BLEU Score")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def __bleu_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the BLEU score per datapoint.
        """
        return [sentence_bleu([truth.split()], pred.split()) for pred, truth in zip(predictions, truths)]

    @staticmethod
    def plot_meteor(predictions: List[str], truths: List[str]):
        """
        Plots the METEOR score per datapoint.
        """
        scores = ScoringMetricsVisualization.__meteor_per_point(predictions, truths)
        plt.hist(scores, bins=20, edgecolor='k', alpha=0.7)
        plt.title("METEOR Scores")
        plt.xlabel("METEOR Score")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def __meteor_per_point(predictions: List[str], truths: List[str]) -> List[float]:
        """
        Computes the METEOR score per datapoint.
        """
        return [meteor_score([truth], pred) for pred, truth in zip(predictions, truths)]
