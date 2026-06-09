from __future__ import annotations


def robustness_from_scores(scores: list[int | float | None]) -> dict:
    clean = [int(score) for score in scores if score is not None]
    n = len(clean)
    k = sum(clean)
    majority_threshold = n // 2 + 1
    return {
        "n": n,
        "k_correct": k,
        "at_least_once_robust": 1 if k >= 1 and n > 0 else 0,
        "majority_robust": 1 if n > 0 and k >= majority_threshold else 0,
        "all_robust": 1 if n > 0 and k == n else 0,
    }
