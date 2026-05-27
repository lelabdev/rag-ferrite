use anyhow::Result;
use rag_engine::api::hybrid_search;

use crate::embedding::EmbeddingProvider;
use crate::types::{BenchmarkDetail, BenchmarkResult, GoldenEntry};

use super::search::search_hybrid;

/// Run a benchmark against a golden dataset.
/// For each entry, queries the engine and checks if expected source_ids appear in top results.
pub async fn run_benchmark(
    embedder: &EmbeddingProvider,
    entries: Vec<GoldenEntry>,
    collection: Option<String>,
    limit: usize,
) -> Result<BenchmarkResult> {
    let mut details = Vec::with_capacity(entries.len());
    let mut total_score = 0.0;
    let mut hits = 0usize;

    let filter = collection.map(|c| hybrid_search::SearchFilter {
        source_ids: None,
        metadata_like: None,
        collection_id: Some(c),
    });

    for entry in &entries {
        let results = match search_hybrid(embedder, &entry.question, limit, filter.clone()).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Benchmark query '{}' failed: {}", entry.question, e);
                vec![]
            }
        };

        // Collect unique source_ids from results
        let found_ids: Vec<i64> = results
            .iter()
            .map(|r| r.source_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Score: fraction of expected source_ids that appear in found_ids
        let matched = entry.relevant_source_ids.iter().filter(|id| found_ids.contains(id)).count();
        let score = if entry.relevant_source_ids.is_empty() {
            0.0
        } else {
            matched as f64 / entry.relevant_source_ids.len() as f64
        };
        let is_hit = matched > 0;

        if is_hit {
            hits += 1;
        }
        total_score += score;

        details.push(BenchmarkDetail {
            query: entry.question.clone(),
            expected_source_ids: entry.relevant_source_ids.clone(),
            found_source_ids: found_ids,
            score,
            is_hit,
        });
    }

    let total_queries = entries.len();
    let misses = total_queries - hits;
    let avg_score = if total_queries > 0 {
        total_score / total_queries as f64
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        total_queries,
        hits,
        misses,
        avg_score,
        details,
    })
}

/// Graph data for document similarity visualization.
pub fn get_graph_data(
    collection: Option<&str>,
    threshold: f32,
    max_edges: usize,
) -> Result<crate::types::GraphData> {
    let conn = super::get_conn()?;

    // 1. Get sources, optionally filtered by collection
    let sources: Vec<(i64, Option<String>, String, i32)> = if let Some(coll) = collection {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s WHERE s.collection_id = ?1 ORDER BY s.id",
        )?;
        stmt.query_map(rusqlite::params![coll], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    } else {
        let mut stmt = conn.prepare(
            "SELECT s.id, s.name, s.collection_id, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) \
             FROM sources s ORDER BY s.id",
        )?;
        stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?.filter_map(|r| r.ok()).collect()
    };

    if sources.is_empty() {
        return Ok(crate::types::GraphData {
            nodes: vec![],
            edges: vec![],
        });
    }

    // Build nodes
    let nodes: Vec<crate::types::GraphNode> = sources
        .iter()
        .map(|(id, name, collection_id, chunk_count)| crate::types::GraphNode {
            id: *id,
            name: name.clone().unwrap_or_else(|| format!("doc_{}", id)),
            collection: collection_id.clone(),
            chunk_count: *chunk_count,
        })
        .collect();

    let source_ids: Vec<i64> = sources.iter().map(|(id, _, _, _)| *id).collect();

    // 2. Load chunk embeddings per source and compute centroids
    let mut centroids: std::collections::HashMap<i64, Vec<f32>> = std::collections::HashMap::new();

    for source_id in &source_ids {
        let mut stmt = conn.prepare(
            "SELECT embedding FROM chunks WHERE source_id = ?1",
        )?;
        let embeddings: Vec<Vec<f32>> = stmt
            .query_map(rusqlite::params![source_id], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(decode_f32_embedding(&blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|v| if v.is_empty() { None } else { Some(v) })
            .collect();

        if embeddings.is_empty() {
            continue;
        }

        // Compute centroid (average of all chunk embeddings)
        let dims = embeddings[0].len();
        let mut centroid = vec![0.0f32; dims];
        let count = embeddings.len() as f32;
        for emb in &embeddings {
            for (i, val) in emb.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in centroid.iter_mut() {
            *val /= count;
        }
        centroids.insert(*source_id, centroid);
    }

    // 3. Compute pairwise cosine similarity
    let mut edges: Vec<crate::types::GraphEdge> = Vec::new();
    let ids_with_centroids: Vec<i64> = source_ids
        .iter()
        .filter(|id| centroids.contains_key(id))
        .copied()
        .collect();

    for i in 0..ids_with_centroids.len() {
        for j in (i + 1)..ids_with_centroids.len() {
            let id_a = ids_with_centroids[i];
            let id_b = ids_with_centroids[j];
            let a = &centroids[&id_a];
            let b = &centroids[&id_b];
            let sim = cosine_similarity(a, b);
            if sim >= threshold {
                edges.push(crate::types::GraphEdge {
                    source: id_a,
                    target: id_b,
                    similarity: (sim * 10000.0).round() / 10000.0, // 4 decimal places
                });
            }
        }
    }

    // 4. Sort by similarity desc, keep max_edges
    edges.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges.truncate(max_edges);

    Ok(crate::types::GraphData { nodes, edges })
}

/// Decode a BLOB of native-endian f32 bytes into a Vec<f32>.
fn decode_f32_embedding(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 {
        return Vec::new();
    }
    blob.chunks(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Compute cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
