from typing import List
from nltk import download
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import wordpunct_tokenize
import sacrebleu

class ScoringMetrics:
    """
    A class that provides various scoring metrics for evaluating predictions against ground truths.
    Includes Exact Match (EM), Cover Exact Match (Cover-EM), F1 Score, BLEU, and METEOR metrics.
    """

    @staticmethod
    def exact_match(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the Exact Match (EM) score.
        EM measures the proportion of predicted answers that exactly match the ground truth answers.

        Args:
            predictions (List[str]): List of predicted answers.
            truths (List[str]): List of ground truth answers.

        Returns:
            float: Exact Match score as a proportion of matches.
        """
        # Count the number of exact matches between predictions and truths
        matches = sum(1 for pred, truth in zip(predictions, truths) if pred.lower() == truth.lower())
        # Return the proportion of exact matches
        return matches / len(truths) if truths else 0.0

    @staticmethod
    def cover_exact_match(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the Cover Exact Match (Cover-EM) score with tokenization using NLTK.
        Cover-EM measures the proportion of predicted answers that contain the ground truth as a substring.
        Tokens are used for comparison to account for different word forms and flexibility.

        Args:
            predictions (List[str]): List of predicted answers.
            truths (List[str]): List of ground truth answers.

        Returns:
            float: Cover Exact Match score as a proportion of matches.
        """
        # Ensure NLTK resources are downloaded
        ScoringMetrics._ensure_nltk_resources()

        # Count the number of predictions that contain the truth as a substring
        matches = sum(
            1 for pred, truth in zip(predictions, truths)
            if all(token in wordpunct_tokenize(pred.lower()) for token in wordpunct_tokenize(truth.lower()))
        )

        # Return the proportion of matches
        return matches / len(truths) if truths else 0.0

    @staticmethod
    def f1_score(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the average F1 score for each sentence pair.

        Args:
            predictions (List[str]): List of predicted answers.
            truths (List[str]): List of ground truth answers.

        Returns:
            float: Average F1 score across all sentence pairs.
        """
        f1_scores = []
        for pred, truth in zip(predictions, truths):
            pred_tokens = pred.split()
            truth_tokens = truth.split()
            true_positives = len(set(pred_tokens) & set(truth_tokens))
            precision = true_positives / len(pred_tokens) if pred_tokens else 0.0
            recall = true_positives / len(truth_tokens) if truth_tokens else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    @staticmethod
    def bleu(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the average BLEU score for each sentence pair using sacrebleu.
        BLEU measures the similarity between a predicted sentence and a reference sentence using n-grams.

        Args:
            predictions (List[str]): List of predicted answers.
            truths (List[str]): List of ground truth answers.

        Returns:
            float: Average BLEU score across all sentence pairs.
        """
         # Compute BLEU score using sacrebleu
        bleu_score = sacrebleu.corpus_bleu(predictions, [truths], lowercase=True)
        return bleu_score.score / 100  # sacrebleu returns BLEU in percentage

    @staticmethod
    def meteor(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the average METEOR score for each sentence pair.
        METEOR evaluates a predicted sentence against a reference sentence considering synonyms and word stems.

        Args:
            predictions (List[str]): List of predicted answers.
            truths (List[str]): List of ground truth answers.

        Returns:
            float: Average METEOR score across all sentence pairs.
        """
        # Ensure NLTK resources are downloaded
        ScoringMetrics._ensure_nltk_resources()

        scores = [
            meteor_score([truth.lower().split()], pred.lower().split())
            for pred, truth in zip(predictions, truths)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _ensure_nltk_resources():
        """
        Ensures that the necessary NLTK resources are downloaded.
        This is a private method.
        """
        try:
            print("Checking NLTK resources...")
            download('wordnet', quiet=True)  # Download WordNet
            print("WordNet resource is available.")
            download('punkt', quiet=True)   # Download Punkt tokenizer (if not already present)
            print("Punkt tokenizer is available.")
        except Exception as e:
            print(f"Error downloading NLTK resources: {e}")
