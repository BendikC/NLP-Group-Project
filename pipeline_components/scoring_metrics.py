from typing import List
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score

class ScoringMetrics:
    @staticmethod
    def exact_match(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the Exact Match (EM) score.
        EM measures the proportion of predicted answers that exactly match the ground truth answers.
        """
        # Count the number of exact matches between predictions and truths
        matches = sum(1 for pred, truth in zip(predictions, truths) if pred == truth)
        # Return the proportion of exact matches
        return matches / len(truths) if truths else 0.0

    @staticmethod
    def cover_exact_match(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the Cover Exact Match (cover-EM) score.
        Cover-EM measures the proportion of predicted answers that contain the ground truth as a substring.
        """
        # Count the number of predictions that contain the truth as a substring
        matches = sum(1 for pred, truth in zip(predictions, truths) if truth in pred)
        # Return the proportion of matches
        return matches / len(truths) if truths else 0.0

    @staticmethod
    def f1_score(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the F1 score.
        The F1 score balances precision and recall, allowing partial matches between predictions and truths.
        """
        # Tokenize and calculate F1 score using sklearn
        y_pred = [pred.split() for pred in predictions]
        y_true = [truth.split() for truth in truths]

        # Flatten tokenized lists and calculate macro F1
        all_preds = [token for tokens in y_pred for token in tokens]
        all_truths = [token for tokens in y_true for token in tokens]

        return f1_score(all_truths, all_preds, average='macro')

    @staticmethod
    def bleu(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the BLEU score.
        BLEU measures the n-gram overlap between predictions and truths, focusing on precision.
        """
        # Calculate BLEU score for each prediction-truth pair
        scores = [sentence_bleu([truth.split()], pred.split()) for pred, truth in zip(predictions, truths)]
        # Return the average BLEU score
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def meteor(predictions: List[str], truths: List[str]) -> float:
        """
        Computes the METEOR score.
        METEOR accounts for precision, recall, and semantic similarity to evaluate predictions.
        """
        # Calculate METEOR score for each prediction-truth pair
        scores = [meteor_score([truth], pred) for pred, truth in zip(predictions, truths)]
        # Return the average METEOR score
        return sum(scores) / len(scores) if scores else 0.0
