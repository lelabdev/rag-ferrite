use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Ground-truth relevance judgment for a single query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceJudgment {
    pub query: String,
    pub relevant_doc_ids: Vec<i64>,
}

/// Per-query evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub query: String,
    pub precision: f64,
    pub recall: f64,
    pub ndcg: f64,
    pub retrieved_ids: Vec<i64>,
    pub relevant_retrieved: usize,
}

/// Aggregated evaluation result across all test queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub total_queries: usize,
    pub avg_precision: f64,
    pub avg_recall: f64,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub per_query: Vec<QueryMetrics>,
}

/// Compute precision: |relevant ∩ retrieved| / |retrieved|.
///
/// Returns 0.0 when nothing was retrieved.
pub fn compute_precision(retrieved: &[i64], relevant: &[i64]) -> f64 {
    if retrieved.is_empty() {
        return 0.0;
    }
    let rel_set: HashSet<i64> = relevant.iter().copied().collect();
    let hits = retrieved.iter().filter(|id| rel_set.contains(id)).count() as f64;
    hits / retrieved.len() as f64
}

/// Compute recall: |relevant ∩ retrieved| / |relevant|.
///
/// Returns 1.0 when there are no relevant documents (nothing to miss).
pub fn compute_recall(retrieved: &[i64], relevant: &[i64]) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    let rel_set: HashSet<i64> = relevant.iter().copied().collect();
    let hits = retrieved.iter().filter(|id| rel_set.contains(id)).count() as f64;
    hits / relevant.len() as f64
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain).
///
/// Uses binary relevance (1 if relevant, 0 otherwise).
/// Returns 0.0 when there are no relevant documents.
pub fn compute_ndcg(retrieved: &[i64], relevant: &[i64], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let rel_set: HashSet<i64> = relevant.iter().copied().collect();

    // DCG@k
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc_id)| {
            let gain = if rel_set.contains(doc_id) { 1.0 } else { 0.0 };
            gain / (i as f64 + 2.0).ln_1p() // log2(i+2) — position is 1-indexed
        })
        .sum();

    // Ideal DCG@k: all relevant docs at the top positions
    let ideal_gains: Vec<f64> = relevant.iter().map(|_| 1.0).collect();
    let idcg: f64 = ideal_gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &gain)| gain / (i as f64 + 2.0).ln_1p())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Compute MRR (Mean Reciprocal Rank): 1 / rank of the first relevant result.
///
/// Returns 0.0 when no relevant document is found in the retrieved list.
pub fn compute_mrr(retrieved: &[i64], relevant: &[i64]) -> f64 {
    if relevant.is_empty() || retrieved.is_empty() {
        return 0.0;
    }
    let rel_set: HashSet<i64> = relevant.iter().copied().collect();
    for (i, doc_id) in retrieved.iter().enumerate() {
        if rel_set.contains(doc_id) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Count how many retrieved documents are in the relevant set.
pub fn count_relevant_retrieved(retrieved: &[i64], relevant: &[i64]) -> usize {
    let rel_set: HashSet<i64> = relevant.iter().copied().collect();
    retrieved.iter().filter(|id| rel_set.contains(id)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_all_relevant() {
        let retrieved = vec![1, 2, 3];
        let relevant = vec![1, 2, 3];
        assert!((compute_precision(&retrieved, &relevant) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_half_relevant() {
        let retrieved = vec![1, 2, 3, 4];
        let relevant = vec![1, 3];
        assert!((compute_precision(&retrieved, &relevant) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_precision_empty_retrieved() {
        assert!((compute_precision(&[], &[1, 2]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_perfect() {
        let retrieved = vec![1, 2, 3, 4];
        let relevant = vec![1, 3];
        assert!((compute_recall(&retrieved, &relevant) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_half() {
        let retrieved = vec![1, 2];
        let relevant = vec![1, 3, 5];
        let r = compute_recall(&retrieved, &relevant);
        assert!((r - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_empty_relevant() {
        assert!((compute_recall(&[1, 2], &[]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_perfect() {
        let retrieved = vec![1, 2, 3];
        let relevant = vec![1, 2, 3];
        assert!((compute_ndcg(&retrieved, &relevant, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_none_relevant() {
        let retrieved = vec![4, 5, 6];
        let relevant = vec![1, 2, 3];
        assert!((compute_ndcg(&retrieved, &relevant, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg_at_k() {
        // k=1, first doc is relevant → NDCG should be 1.0
        let retrieved = vec![1, 4, 5];
        let relevant = vec![1, 2, 3];
        assert!((compute_ndcg(&retrieved, &relevant, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_first_position() {
        let retrieved = vec![1, 2, 3];
        let relevant = vec![1];
        assert!((compute_mrr(&retrieved, &relevant) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_third_position() {
        let retrieved = vec![4, 5, 1];
        let relevant = vec![1];
        assert!((compute_mrr(&retrieved, &relevant) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_mrr_not_found() {
        let retrieved = vec![4, 5, 6];
        let relevant = vec![1];
        assert!((compute_mrr(&retrieved, &relevant) - 0.0).abs() < 1e-9);
    }
}
