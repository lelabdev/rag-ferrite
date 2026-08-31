#![cfg(test)]
#![allow(clippy::useless_vec)]
/// Integration test for storage module (replaces rag_engine).
/// Tests brute-force cosine + FTS5 BM25 + RRF fusion without any external API.
use rusqlite::Connection;

/// Helper: create a fresh in-memory DB with schema + FTS5
fn setup_test_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            metadata TEXT,
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            name TEXT,
            status TEXT DEFAULT 'completed',
            collection_id TEXT NOT NULL DEFAULT '__default__'
        );",
    )
    .unwrap();

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            collection_id TEXT NOT NULL DEFAULT '__default__',
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_pos INTEGER NOT NULL,
            end_pos INTEGER NOT NULL,
            chunk_type TEXT DEFAULT 'general',
            embedding BLOB,
            embedding_i8 BLOB,
            embedding_scale REAL
        );",
    )
    .unwrap();

    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            chunk_id UNINDEXED,
            tokenize='porter unicode61'
        );",
    )
    .unwrap();

    conn
}

/// Helper: convert Vec<f32> to bytes for SQLite storage
fn emb_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// Helper: cosine similarity (same formula as storage/sqlite.rs)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)) as f64
}

/// Helper: RRF score (same formula as storage/sqlite.rs)
fn rrf_score(rank: usize, k: u32) -> f64 {
    1.0 / (k as f64 + rank as f64)
}

#[test]
fn test_fts5_bm25_search() {
    let conn = setup_test_db();

    // Insert test chunks
    let docs = vec![
        (
            1,
            "Rust is a systems programming language focused on safety and speed",
        ),
        (
            2,
            "Python is a popular language for machine learning and data science",
        ),
        (
            3,
            "The Tokyo metropolitan area is the most populous in the world",
        ),
    ];

    for (id, content) in &docs {
        conn.execute(
            "INSERT INTO chunks (id, source_id, chunk_index, content, start_pos, end_pos) VALUES (?1, 1, ?2, ?3, 0, 100)",
            rusqlite::params![id, id - 1, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
            rusqlite::params![content, id],
        )
        .unwrap();
    }

    // Search for "Rust programming"
    let mut stmt = conn
        .prepare(
            "SELECT chunk_id, bm25(chunks_fts) as score
         FROM chunks_fts
         WHERE chunks_fts MATCH 'Rust programming'
         ORDER BY score
         LIMIT 5",
        )
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, -row.get::<_, f64>(1)?)) // negate to make positive
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        !results.is_empty(),
        "BM25 should return results for 'Rust programming'"
    );
    assert_eq!(results[0].0, 1, "Chunk about Rust should rank first");
    println!(
        "✅ BM25 search: chunk {} scored {:.6}",
        results[0].0, results[0].1
    );
}

#[test]
fn test_cosine_similarity_brute_force() {
    let query_embedding = vec![1.0, 0.0, 0.0, 0.0]; // query = "Rust"

    let chunk_embeddings = vec![
        (1_i64, vec![0.9, 0.1, 0.0, 0.0]), // similar to Rust
        (2_i64, vec![0.0, 0.9, 0.1, 0.0]), // different topic (Python)
        (3_i64, vec![0.1, 0.0, 0.0, 0.9]), // different topic (Tokyo)
    ];

    let mut scored: Vec<(i64, f64)> = chunk_embeddings
        .iter()
        .map(|(id, emb)| (*id, cosine_similarity(&query_embedding, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    assert_eq!(
        scored[0].0, 1,
        "Chunk 1 (Rust-like) should have highest cosine similarity"
    );
    assert!(
        scored[0].1 > 0.9,
        "Similarity should be high: {:.4}",
        scored[0].1
    );
    println!(
        "✅ Cosine similarity: chunk {} scored {:.4}",
        scored[0].0, scored[0].1
    );
}

#[test]
fn test_rrf_fusion() {
    // Simulate vector and BM25 rankings
    // Vector search ranks: chunk1 > chunk2 > chunk3
    // BM25 search ranks: chunk2 > chunk1 > chunk3
    let vector_results = vec![(1_i64, 0.95), (2_i64, 0.80), (3_i64, 0.50)];
    let bm25_results = vec![(2_i64, 3.5), (1_i64, 2.8), (3_i64, 1.2)];

    let k = 60_u32;

    let mut vector_ranks = std::collections::HashMap::new();
    for (rank, (id, _)) in vector_results.iter().enumerate() {
        vector_ranks.insert(*id, rank + 1);
    }

    let mut bm25_ranks = std::collections::HashMap::new();
    for (rank, (id, _)) in bm25_results.iter().enumerate() {
        bm25_ranks.insert(*id, rank + 1);
    }

    let mut rrf_scores: Vec<(i64, f64)> = vec![1, 2, 3]
        .iter()
        .map(|id| {
            let vec_rank = vector_ranks.get(id).copied();
            let bm25_rank = bm25_ranks.get(id).copied();
            let mut score = 0.0;
            if let Some(r) = vec_rank {
                score += rrf_score(r, k);
            }
            if let Some(r) = bm25_rank {
                score += rrf_score(r, k);
            }
            (*id, score)
        })
        .collect();

    rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Chunk 1: rank 1 in vector + rank 2 in BM25 = 1/61 + 1/62 ≈ 0.0325
    // Chunk 2: rank 2 in vector + rank 1 in BM25 = 1/62 + 1/61 ≈ 0.0325
    // Both should be very close, chunk 1 and 2 should be top 2
    assert!(
        rrf_scores[0].0 == 1 || rrf_scores[0].0 == 2,
        "Top result should be chunk 1 or 2"
    );
    assert!(rrf_scores[2].0 == 3, "Chunk 3 should be last");

    println!("✅ RRF fusion:");
    for (id, score) in &rrf_scores {
        println!("   chunk {}: {:.6}", id, score);
    }
}

#[test]
fn test_full_hybrid_pipeline() {
    let conn = setup_test_db();

    // Insert 5 chunks with embeddings
    // Chunks 1-2 about Rust, 3-4 about Python, 5 about Tokyo
    let chunks = vec![
        (
            1,
            1,
            "Rust programming language safety speed concurrency",
            vec![0.9, 0.1, 0.0],
        ),
        (
            2,
            1,
            "Rust memory safety borrow checker compile time",
            vec![0.85, 0.15, 0.0],
        ),
        (
            3,
            2,
            "Python machine learning data science numpy",
            vec![0.1, 0.9, 0.0],
        ),
        (
            4,
            2,
            "Python pandas tensorflow neural networks",
            vec![0.0, 0.85, 0.1],
        ),
        (
            5,
            3,
            "Tokyo is the capital of Japan metropolitan area",
            vec![0.0, 0.0, 0.95],
        ),
    ];

    for (id, source_id, content, emb) in &chunks {
        conn.execute(
            "INSERT INTO chunks (id, source_id, chunk_index, content, start_pos, end_pos, embedding)
             VALUES (?1, ?2, ?3, ?4, 0, 100, ?5)",
            rusqlite::params![id, source_id, id - 1, content, emb_to_bytes(emb)],
        ).unwrap();
        conn.execute(
            "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
            rusqlite::params![content, id],
        )
        .unwrap();
    }

    // Simulate query "Rust programming" with embedding [0.9, 0.1, 0.0]
    let query_embedding = vec![0.9_f32, 0.1, 0.0];
    let query_text = "Rust programming";

    // 1. Vector search (brute-force cosine)
    let mut stmt = conn.prepare("SELECT id, embedding FROM chunks").unwrap();
    let vector_results: Vec<(i64, f64)> = stmt
        .query_map([], |row| {
            let id: i64 = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            let (chunks, _) = bytes.as_chunks::<4>();
            let emb: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
            Ok((id, emb))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .map(|(id, emb)| (id, cosine_similarity(&query_embedding, &emb)))
        .collect();

    let mut vector_sorted = vector_results.clone();
    vector_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    vector_sorted.truncate(5);

    assert_eq!(
        vector_sorted[0].0, 1,
        "Vector search: chunk 1 should be first"
    );
    assert_eq!(
        vector_sorted[1].0, 2,
        "Vector search: chunk 2 should be second"
    );

    // 2. BM25 search (FTS5)
    let mut stmt = conn
        .prepare(
            "SELECT chunk_id, bm25(chunks_fts) as score
         FROM chunks_fts
         WHERE chunks_fts MATCH ?1
         ORDER BY score
         LIMIT 5",
        )
        .unwrap();
    let bm25_results: Vec<(i64, f64)> = stmt
        .query_map([query_text], |row| {
            Ok((row.get(0)?, -row.get::<_, f64>(1)?))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(!bm25_results.is_empty(), "BM25 should return results");
    // BM25 should rank Rust chunks high
    let bm25_chunk_ids: Vec<i64> = bm25_results.iter().map(|(id, _)| *id).collect();
    assert!(
        bm25_chunk_ids.contains(&1) || bm25_chunk_ids.contains(&2),
        "BM25 should find Rust chunks"
    );

    // 3. RRF fusion
    let k = 60_u32;
    let mut vector_ranks = std::collections::HashMap::new();
    for (rank, (id, _)) in vector_sorted.iter().enumerate() {
        vector_ranks.insert(*id, rank + 1);
    }
    let mut bm25_ranks = std::collections::HashMap::new();
    for (rank, (id, _)) in bm25_results.iter().enumerate() {
        bm25_ranks.insert(*id, rank + 1);
    }

    let mut all_ids: Vec<i64> = vector_ranks
        .keys()
        .chain(bm25_ranks.keys())
        .copied()
        .collect();
    all_ids.sort();
    all_ids.dedup();

    let mut rrf_scores: Vec<(i64, f64, u32, u32)> = all_ids
        .iter()
        .map(|id| {
            let vr = vector_ranks.get(id).copied();
            let br = bm25_ranks.get(id).copied();
            let mut score = 0.0;
            if let Some(r) = vr {
                score += rrf_score(r, k);
            }
            if let Some(r) = br {
                score += rrf_score(r, k);
            }
            (*id, score, vr.unwrap_or(0) as u32, br.unwrap_or(0) as u32)
        })
        .collect();
    rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top results should be Rust chunks (1 or 2)
    assert!(
        rrf_scores[0].0 == 1 || rrf_scores[0].0 == 2,
        "Top result should be a Rust chunk, got chunk {}",
        rrf_scores[0].0
    );

    println!("✅ Full hybrid pipeline (query: 'Rust programming'):");
    for (id, score, vr, br) in rrf_scores.iter().take(3) {
        println!(
            "   chunk {}: RRF={:.6} (vec_rank={}, bm25_rank={})",
            id, score, vr, br
        );
    }
}

#[test]
fn test_fts5_multilingual() {
    let conn = setup_test_db();

    let docs = vec![
        (1, "Tokyo est la capitale du Japon"), // French
        (2, "Tokyo is the capital of Japan"),  // English
        (3, "東京は日本の首都です"),           // Japanese
    ];

    for (id, content) in &docs {
        conn.execute(
            "INSERT INTO chunks (id, source_id, chunk_index, content, start_pos, end_pos) VALUES (?1, 1, ?2, ?3, 0, 50)",
            rusqlite::params![id, id - 1, content],
        ).unwrap();
        conn.execute(
            "INSERT INTO chunks_fts (content, chunk_id) VALUES (?1, ?2)",
            rusqlite::params![content, id],
        )
        .unwrap();
    }

    // Search for "Tokyo" — should match all 3
    let mut stmt = conn
        .prepare("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'Tokyo' ORDER BY chunk_id")
        .unwrap();
    let results: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        results.len() >= 2,
        "FTS5 should find at least 2 results for 'Tokyo', got {}",
        results.len()
    );
    println!(
        "✅ FTS5 multilingual: found {} results for 'Tokyo'",
        results.len()
    );
}

#[test]
fn test_fts5_unicode_tokenizer() {
    let conn = setup_test_db();

    conn.execute(
        "INSERT INTO chunks (id, source_id, chunk_index, content, start_pos, end_pos) VALUES (1, 1, 0, '富士山は日本最高峰の山です', 0, 50)",
        [],
    ).unwrap();
    conn.execute(
        "INSERT INTO chunks_fts (content, chunk_id) VALUES ('富士山は日本最高峰の山です', 1)",
        [],
    )
    .unwrap();

    let mut stmt = conn
        .prepare(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH '富士山は日本最高峰の山です'",
        )
        .unwrap();
    let results: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();

    assert_eq!(
        results,
        vec![1],
        "FTS5 unicode61 tokenizer should preserve Japanese text as a searchable token"
    );
    println!("✅ FTS5 unicode tokenizer: exact Japanese search works");
}
