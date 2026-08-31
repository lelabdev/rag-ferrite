use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct EmbeddingProvider {
    provider: String,
    model: String,
    dimensions: Option<usize>,
    detected_dimensions: std::sync::Arc<std::sync::OnceLock<usize>>,
    api_key: Option<String>,
    base_url: Option<String>,
    client: reqwest::Client,
    batch_size: usize,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

impl EmbeddingProvider {
    pub fn new(
        provider: String,
        model: String,
        dimensions: Option<usize>,
        api_key: Option<String>,
        base_url: Option<String>,
        batch_size: usize,
    ) -> Self {
        let api_key = api_key
            .or_else(|| std::env::var("EMBEDDING_API_KEY").ok())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());
        Self {
            provider,
            model,
            dimensions,
            detected_dimensions: std::sync::Arc::new(std::sync::OnceLock::new()),
            api_key,
            base_url,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap(),
            batch_size,
        }
    }

    /// Get embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let vecs = self.embed_batch(&[text.to_string()]).await?;
        vecs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    /// Get embeddings for multiple texts (batch)
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let vecs = match self.provider.as_str() {
            "openai" => self.embed_openai(texts).await,
            "cohere" => self.embed_cohere(texts).await,
            "ollama" => self.embed_ollama(texts).await,
            _ => Err(anyhow!("Unknown embedding provider: {}", self.provider)),
        }?;

        Ok(vecs)
    }

    pub fn validate_batch(&self, expected_count: usize, embeddings: &[Vec<f32>]) -> Result<()> {
        self.validate_vectors(expected_count, embeddings)
    }

    fn validate_vectors(&self, expected_count: usize, embeddings: &[Vec<f32>]) -> Result<()> {
        if embeddings.len() != expected_count {
            return Err(anyhow!(
                "Embedding provider returned {} vectors for {} texts",
                embeddings.len(),
                expected_count
            ));
        }

        if embeddings.is_empty() {
            return Ok(());
        }

        let expected_dimensions = self
            .dimensions
            .or_else(|| self.detected_dimensions.get().copied());
        let observed_dimensions = embeddings[0].len();
        if observed_dimensions == 0 {
            return Err(anyhow!("Embedding vectors must have a non-zero dimension"));
        }
        let expected_dimensions = expected_dimensions.unwrap_or(observed_dimensions);

        for (index, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != expected_dimensions {
                return Err(anyhow!(
                    "Embedding {} has dimension {}, expected {}",
                    index,
                    embedding.len(),
                    expected_dimensions
                ));
            }
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(anyhow!("Embedding {} contains non-finite values", index));
            }
        }

        if self.dimensions.is_none() && self.detected_dimensions.get().is_none() {
            if self.detected_dimensions.set(observed_dimensions).is_ok() {
                tracing::info!(
                    "Auto-detected embedding dimensions: {}",
                    observed_dimensions
                );
            }
        }

        if let Some(detected) = self.detected_dimensions.get()
            && *detected != observed_dimensions
        {
            return Err(anyhow!(
                "Embedding batch dimension {}, expected detected dimension {}",
                observed_dimensions,
                detected
            ));
        }

        Ok(())
    }

    async fn embed_openai(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            anyhow!("API key required for OpenAI. Set EMBEDDING_API_KEY env or config.")
        })?;

        let base = self
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        let url = format!("{}/embeddings", base);

        let batch_size = if self.batch_size > 0 {
            self.batch_size
        } else {
            20
        };
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for (i, chunk) in texts.chunks(batch_size).enumerate() {
            let total_batches = texts.len().div_ceil(batch_size);
            tracing::info!(
                "Embedding batch {}/{} ({} texts)",
                i + 1,
                total_batches,
                chunk.len()
            );

            let body = EmbeddingRequest {
                model: self.model.clone(),
                input: chunk.to_vec(),
                dimensions: self.dimensions,
            };

            let mut attempt = 0;
            let max_retries = 3;
            let resp = loop {
                attempt += 1;
                match self
                    .client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(r) => break r,
                    Err(e) if attempt < max_retries => {
                        tracing::warn!(
                            "Embedding attempt {}/{} failed: {} — retrying in 2s",
                            attempt,
                            max_retries,
                            e
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                    Err(e) => {
                        tracing::error!("Embedding failed after {} attempts: {}", max_retries, e);
                        return Err(anyhow!("Embedding request failed: {}", e));
                    }
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                tracing::error!("Embedding API error {}: {}", status, text);
                return Err(anyhow!("OpenAI API error {}: {}", status, text));
            }

            let data: EmbeddingResponse = resp.json().await?;
            if data.data.len() != chunk.len() {
                return Err(anyhow!(
                    "OpenAI returned {} vectors for {} texts in sub-batch {}",
                    data.data.len(),
                    chunk.len(),
                    i + 1
                ));
            }
            let mut indexed = Vec::with_capacity(data.data.len());
            for item in data.data {
                if item.index >= chunk.len() {
                    return Err(anyhow!(
                        "OpenAI embedding index {} is out of range for {} texts",
                        item.index,
                        chunk.len()
                    ));
                }
                indexed.push((item.index, item.embedding));
            }
            indexed.sort_unstable_by_key(|(index, _)| *index);
            if indexed
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| *actual != expected)
            {
                return Err(anyhow!(
                    "OpenAI embedding indexes must map each sub-batch text exactly once"
                ));
            }
            let batch_embeddings: Vec<Vec<f32>> = indexed
                .into_iter()
                .map(|(_, embedding)| embedding)
                .collect();
            self.validate_vectors(chunk.len(), &batch_embeddings)?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    async fn embed_cohere(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            anyhow!("API key required for Cohere. Set EMBEDDING_API_KEY env or config.")
        })?;

        let base = self
            .base_url
            .as_deref()
            .unwrap_or("https://api.cohere.ai/v1");
        let url = format!("{}/embed", base);

        #[derive(Debug, Serialize)]
        #[serde(rename_all = "snake_case")]
        enum CohereInputType {
            SearchDocument,
            /// Reserved for future query-time embeddings.
            #[allow(dead_code)]
            SearchQuery,
        }

        #[derive(Debug, Serialize)]
        struct CohereRequest {
            model: String,
            texts: Vec<String>,
            input_type: CohereInputType,
            embedding_types: Option<Vec<String>>,
        }

        let body = CohereRequest {
            model: self.model.clone(),
            texts: texts.to_vec(),
            input_type: CohereInputType::SearchDocument,
            embedding_types: Some(vec!["float".into()]),
        };

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await?;
            return Err(anyhow!("Cohere API error {}: {}", status, text));
        }

        #[derive(Debug, Deserialize)]
        struct CohereResponse {
            embeddings: CohereEmbeddings,
        }

        #[derive(Debug, Deserialize)]
        struct CohereEmbeddings {
            float: Option<Vec<Vec<f32>>>,
        }

        let data: CohereResponse = resp.json().await?;
        let embeddings = data
            .embeddings
            .float
            .ok_or_else(|| anyhow!("No float embeddings in Cohere response"))?;
        self.validate_vectors(texts.len(), &embeddings)?;
        Ok(embeddings)
    }

    async fn embed_ollama(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let base = self.base_url.as_deref().unwrap_or("http://localhost:11434");

        #[derive(Debug, Serialize)]
        struct OllamaRequest {
            model: String,
            input: Vec<String>,
        }

        #[derive(Debug, Deserialize)]
        struct OllamaResponse {
            embeddings: Vec<Vec<f32>>,
        }

        // Ollama batch can be large, so we limit to 20 texts per request
        // to avoid timeout and memory issues
        let mut results = Vec::with_capacity(texts.len());
        let batch_size = if self.batch_size > 0 {
            self.batch_size
        } else {
            20
        };

        for batch in texts.chunks(batch_size) {
            let url = format!("{}/api/embed", base);

            let body = OllamaRequest {
                model: self.model.clone(),
                input: batch.to_vec(),
            };

            let resp = self.client.post(&url).json(&body).send().await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await?;
                return Err(anyhow!("Ollama API error {}: {}", status, text));
            }

            let data: OllamaResponse = resp.json().await?;
            self.validate_vectors(batch.len(), &data.embeddings)?;
            results.extend(data.embeddings);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingProvider;

    fn provider(dimensions: Option<usize>) -> EmbeddingProvider {
        EmbeddingProvider::new(
            "test".into(),
            "test-model".into(),
            dimensions,
            None,
            None,
            1,
        )
    }

    #[test]
    fn rejects_incomplete_batches() {
        let result = provider(Some(2)).validate_batch(2, &[vec![0.0, 1.0]]);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("returned 1 vectors")
        );
    }

    #[test]
    fn rejects_wrong_dimensions() {
        let result = provider(Some(2)).validate_batch(1, &[vec![0.0, 1.0, 2.0]]);
        assert!(result.unwrap_err().to_string().contains("expected 2"));
    }

    #[test]
    fn rejects_non_finite_values() {
        let result = provider(None).validate_batch(1, &[vec![f32::NAN]]);
        assert!(result.unwrap_err().to_string().contains("non-finite"));
    }

    #[test]
    fn rejects_zero_length_vectors() {
        let result = provider(None).validate_batch(1, &[vec![]]);
        assert!(result.unwrap_err().to_string().contains("non-zero"));
    }

    #[test]
    fn keeps_detected_dimension_consistent_across_batches() {
        let provider = provider(None);
        provider.validate_batch(1, &[vec![0.0, 1.0]]).unwrap();
        let result = provider.validate_batch(1, &[vec![0.0, 1.0, 2.0]]);
        assert!(result.unwrap_err().to_string().contains("expected 2"));
    }

    #[test]
    fn invalid_first_batch_does_not_poison_dimension_detection() {
        let provider = provider(None);
        assert!(provider.validate_batch(1, &[vec![]]).is_err());
        provider.validate_batch(1, &[vec![0.0, 1.0]]).unwrap();
        assert!(provider.validate_batch(1, &[vec![0.0, 1.0, 2.0]]).is_err());
    }
}
