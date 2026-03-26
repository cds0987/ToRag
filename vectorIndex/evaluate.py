import numpy as np
import math

def recall_at_k(pred, gt_set, k):
    return int(len(set(pred[:k]) & gt_set) > 0)

def mrr(pred, gt_set):
    for i, p in enumerate(pred):
        if p in gt_set:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(pred, gt_set, k):
    dcg = 0.0
    for i, p in enumerate(pred[:k]):
        if p in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(pred, gt_set, k):
    return len(set(pred[:k]) & gt_set) / k
def hit_rate(pred, gt_set, k):
    return int(len(set(pred[:k]) & gt_set) > 0)
def recall_full(pred, gt_set, k):
    return len(set(pred[:k]) & gt_set) / len(gt_set) if gt_set else 0

def average_precision(pred, gt_set, k):
    hits = 0
    score = 0.0

    for i, p in enumerate(pred[:k]):
        if p in gt_set:
            hits += 1
            score += hits / (i + 1)

    return score / len(gt_set) if gt_set else 0.0


import numpy as np
import pandas as pd


def evaluate_retrieval(
    query_ids,
    retrieved,
    gt,
    Ks=None,
    save_path="taskadaptive_eval.csv",
    verbose=True
):
    """
    query_ids: List[str]
    retrieved: List[List[str]]
    gt: Dict[str, Set[str]]
    Ks: List[int] (optional)
    """

    assert len(query_ids) == len(retrieved), "Mismatch query_ids and retrieved"

    # infer max K
    max_k = max(len(r) for r in retrieved)

    if Ks is None:
      Ks = np.unique(np.logspace(0, np.log10(max_k), num=10, dtype=int)).tolist()

    # filter valid queries
    valid_data = [
        (qid, pred)
        for qid, pred in zip(query_ids, retrieved)
        if len(gt.get(qid, set())) > 0
    ]

    results = []

    for k in Ks:
        recall_hit_list = []
        recall_full_list = []
        precision_list = []
        ndcg_list = []
        ap_list = []
        mrr_list = []

        for qid, pred in valid_data:
            gt_set = gt[qid]

            recall_hit_list.append(recall_at_k(pred, gt_set, k))
            recall_full_list.append(recall_full(pred, gt_set, k))
            precision_list.append(precision_at_k(pred, gt_set, k))
            ndcg_list.append(ndcg_at_k(pred, gt_set, k))
            ap_list.append(average_precision(pred, gt_set, k))

            if k == Ks[-1]:
                mrr_list.append(mrr(pred, gt_set))

        row = {
            "K": k,
            "Recall@K(hit)": np.mean(recall_hit_list),
            "Recall@K(full)": np.mean(recall_full_list),
            "Precision@K": np.mean(precision_list),
            "nDCG@K": np.mean(ndcg_list),
            "MAP@K": np.mean(ap_list),
            "MRR": np.mean(mrr_list) if k == Ks[-1] else None
        }

        results.append(row)

    df = pd.DataFrame(results)

    # display
    if verbose:
        print("\n=== Retrieval Evaluation ===")
        print(df.to_string(index=False, float_format="%.4f"))

    # save
    df.to_csv(save_path, index=False)
    if verbose:
        print(f"\nSaved to {save_path}")

    return df