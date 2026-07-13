"""
Scoring module for evaluating NER predictions.
Implements WER for text and Jaccard similarity for assertions and candidates.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import editdistance  # For WER calculation


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = number of words in reference
    """
    # Tokenize by whitespace and punctuation
    ref_words = re.findall(r'\S+', reference.lower())
    hyp_words = re.findall(r'\S+', hypothesis.lower())

    # Simple word-level edit distance
    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / max(len(ref_words), 1)

    return min(wer, 1.0)  # Cap at 1.0


def jaccard_similarity(set1: List, set2: List) -> float:
    """
    Calculate Jaccard similarity between two sets/lists.
    J = |A ∩ B| / |A ∪ B|
    """
    set1 = set([str(x).lower().strip() for x in set1])
    set2 = set([str(x).lower().strip() for x in set2])

    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def jaccard_for_field(ground_truth: List, prediction: List) -> float:
    """
    Calculate Jaccard similarity for a specific field (assertions or candidates).
    Special cases:
    - If both empty: return 1.0
    - If GT empty but prediction not: return 0.0
    - Otherwise: standard Jaccard
    """
    if len(ground_truth) == 0 and len(prediction) == 0:
        return 1.0
    if len(ground_truth) == 0 and len(prediction) > 0:
        return 0.0

    return jaccard_similarity(ground_truth, prediction)


class NERScorer:
    """Score NER predictions against ground truth."""

    def __init__(self, weights: Dict = None):
        self.weights = weights or {
            'text': 0.3,
            'assertions': 0.3,
            'candidates': 0.4
        }

    def score_sample(self, ground_truth: List[Dict], predictions: List[Dict]) -> Dict:
        """
        Score a single sample.
        Returns individual scores and final weighted score.
        """
        # Calculate text score (WER)
        text_score = self._calculate_text_score(ground_truth, predictions)

        # Calculate assertions score (Jaccard)
        assertions_score = self._calculate_assertions_score(ground_truth, predictions)

        # Calculate candidates score (Jaccard, weighted)
        candidates_score = self._calculate_candidates_score(ground_truth, predictions)

        # Final score
        final_score = (
            self.weights['text'] * text_score +
            self.weights['assertions'] * assertions_score +
            self.weights['candidates'] * candidates_score
        )

        return {
            'text_score': text_score,
            'assertions_score': assertions_score,
            'candidates_score': candidates_score,
            'final_score': final_score
        }

    def _calculate_text_score(self, gt: List[Dict], pred: List[Dict]) -> float:
        """
        Calculate text score using WER.
        For each prediction, find matching GT entity and calculate WER.
        """
        if not gt or not pred:
            return 0.0

        total_wer = 0.0
        matched_count = 0

        for p in pred:
            p_text = p.get('text', '')
            p_type = p.get('type', '')

            # Find best matching GT entity by type
            best_match = None
            best_wer = float('inf')

            for g in gt:
                if g.get('type', '') == p_type:
                    g_text = g.get('text', '')
                    wer = word_error_rate(g_text, p_text)
                    if wer < best_wer:
                        best_wer = wer
                        best_match = g_text

            if best_match is not None:
                total_wer += (1.0 - best_wer)
                matched_count += 1

        if matched_count == 0:
            return 0.0

        return total_wer / max(len(pred), len(gt))

    def _calculate_assertions_score(self, gt: List[Dict], pred: List[Dict]) -> float:
        """Calculate assertions score using Jaccard similarity."""
        if not gt:
            return 1.0 if not pred else 0.0

        total_jaccard = 0.0
        count = 0

        for p in pred:
            p_type = p.get('type', '')
            p_assertions = p.get('assertions', [])

            # Find matching GT by type
            matching_gt = [g for g in gt if g.get('type', '') == p_type]

            if matching_gt:
                gt_assertions = matching_gt[0].get('assertions', [])
                j = jaccard_for_field(gt_assertions, p_assertions)
                total_jaccard += j
                count += 1

        if count == 0:
            return 0.0

        return total_jaccard / count

    def _calculate_candidates_score(self, gt: List[Dict], pred: List[Dict]) -> float:
        """
        Calculate candidates score using Jaccard similarity.
        Weighted by sum of (len(gt_candidates) + 1).
        """
        total_jaccard = 0.0
        total_weight = 0.0

        for p in pred:
            p_type = p.get('type', '')
            p_candidates = p.get('candidates', [])

            # Find matching GT by type
            matching_gt = [g for g in gt if g.get('type', '') == p_type]

            if matching_gt:
                gt_candidates = matching_gt[0].get('candidates', [])
                j = jaccard_for_field(gt_candidates, p_candidates)

                # Weight = len(gt_candidates) + 1
                weight = len(gt_candidates) + 1
                total_jaccard += j * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_jaccard / total_weight

    def score_batch(self, ground_truths: Dict[int, List[Dict]],
                    predictions: Dict[int, List[Dict]]) -> Dict:
        """
        Score a batch of samples.
        ground_truths: {sample_id: [entity_dict, ...]}
        predictions: {sample_id: [entity_dict, ...]}
        """
        sample_scores = []
        total_text = 0.0
        total_assertions = 0.0
        total_candidates = 0.0

        for sample_id in sorted(ground_truths.keys()):
            gt = ground_truths.get(sample_id, [])
            pred = predictions.get(sample_id, [])

            scores = self.score_sample(gt, pred)
            sample_scores.append(scores)

            total_text += scores['text_score']
            total_assertions += scores['assertions_score']
            total_candidates += scores['candidates_score']

        n = len(ground_truths)
        if n == 0:
            return {
                'text_score': 0.0,
                'assertions_score': 0.0,
                'candidates_score': 0.0,
                'final_score': 0.0
            }

        avg_text = total_text / n
        avg_assertions = total_assertions / n

        # Candidates score uses different aggregation
        avg_candidates = total_candidates / n

        final = (
            self.weights['text'] * avg_text +
            self.weights['assertions'] * avg_assertions +
            self.weights['candidates'] * avg_candidates
        )

        return {
            'text_score': avg_text,
            'assertions_score': avg_assertions,
            'candidates_score': avg_candidates,
            'final_score': final,
            'sample_scores': sample_scores
        }


def demo():
    """Demo scoring with example data."""
    gt = [
        {
            "text": "metoprolol 25mg po bid",
            "type": "THUỐC",
            "candidates": ["866436"],
            "assertions": ["isHistorical"],
            "position": [0, 20]
        }
    ]

    pred = [
        {
            "text": "metoprolol 25mg po bid",
            "type": "THUỐC",
            "candidates": ["866436"],
            "assertions": ["isHistorical"],
            "position": [0, 20]
        }
    ]

    scorer = NERScorer()
    score = scorer.score_sample(gt, pred)
    print("Perfect match score:", score)


if __name__ == "__main__":
    demo()
