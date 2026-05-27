use anyhow::Result;
use std::path::Path;
use std::process::Command;

/// Extract text from a file (PDF, TXT, MD)
///
/// For PDF: uses pdftotext (poppler-utils) via subprocess
/// For TXT/MD: reads directly
pub fn extract_text(file_path: &str) -> Result<String> {
    let path = Path::new(file_path);

    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf_text(file_path),
        "txt" | "md" => read_text_file(file_path),
        _ => anyhow::bail!("Unsupported file type: .{} (supported: pdf, txt, md)", ext),
    }
}

/// Extract text from PDF using pdftotext (poppler-utils)
///
/// Requires: poppler-utils package
/// Usage: `apt install poppler-utils` on Debian/Ubuntu
fn extract_pdf_text(pdf_path: &str) -> Result<String> {
    tracing::info!("Extracting text from PDF: {}", pdf_path);

    let output = Command::new("pdftotext")
        .arg(pdf_path)
        .arg("-")  // Write to stdout
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow::anyhow!("pdftotext not found. Install poppler-utils: apt install poppler-utils")
            } else {
                anyhow::anyhow!("Failed to run pdftotext: {}", e)
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pdftotext failed: {}", stderr);
    }

    let text = String::from_utf8_lossy(&output.stdout).to_string();
    let char_count = text.len();

    tracing::info!("Extracted {} chars from PDF ({} KB)", char_count, char_count / 1024);

    // Warn if output suspiciously short
    if char_count < 10_000 {
        tracing::warn!("PDF extraction suspiciously short ({} chars). PDF may be image-based or encrypted.", char_count);
    }

    Ok(text)
}

/// Read text file directly
fn read_text_file(file_path: &str) -> Result<String> {
    let text = std::fs::read_to_string(file_path)?;
    tracing::info!("Read {} chars from {}", text.len(), file_path);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_file() {
        let result = extract_text("/nonexistent/path/to/file.pdf");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"), "Expected 'not found' in error, got: {}", err);
    }

    #[test]
    fn test_non_pdf() {
        // Create a temp file with .txt extension but test .xyz (unsupported)
        let dir = std::env::temp_dir().join("rag_ferrite_test_extractor");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test.xyz");
        std::fs::write(&file_path, "some content").unwrap();

        let path_str = file_path.to_string_lossy().to_string();
        let result = extract_text(&path_str);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unsupported") || err.contains("unsupported"),
            "Expected 'Unsupported' in error, got: {}", err);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extract_txt_file() {
        let dir = std::env::temp_dir().join("rag_ferrite_test_extractor");
        std::fs::create_dir_all(&dir).unwrap();
        let file_path = dir.join("test.txt");
        let content = "Hello, this is a test file.";
        std::fs::write(&file_path, content).unwrap();

        let path_str = file_path.to_string_lossy().to_string();
        let result = extract_text(&path_str);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
