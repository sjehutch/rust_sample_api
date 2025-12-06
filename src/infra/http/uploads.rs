use std::{env, path::Path};
use base64::Engine;

use axum::{
    extract::Multipart,
    http::StatusCode,
    routing::post,
    Json, Router,
    extract::State,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::AppState;

#[derive(Serialize, Deserialize, ToSchema)]
pub struct UploadResponse {
    pub id: String,
    pub path: String,
    pub description: Option<String>,
}

#[allow(dead_code)]
#[derive(ToSchema)]
pub struct UploadBody {
    /// File upload field named "file"
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
}

/// POST /upload
#[utoipa::path(
    post,
    path = "/upload",
    request_body(
        content = UploadBody,
        content_type = "multipart/form-data",
        description = "Upload an image file in form field `file`"
    ),
    responses(
        (status = 201, description = "File uploaded", body = UploadResponse),
        (status = 400, description = "Invalid upload"),
        (status = 500, description = "Failed to store file"),
    ),
    tag = "uploads"
)]
pub async fn upload_image(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<UploadResponse>), (StatusCode, String)> {
    let mut file_bytes = None;
    let mut file_name = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid multipart data: {e}")))? 
    {
        if field.name() == Some("file") {
            file_name = field.file_name().map(|s| s.to_string());
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read file bytes: {e}")))?;
            file_bytes = Some(data);
            break;
        }
    }

    let data = file_bytes.ok_or((StatusCode::BAD_REQUEST, "Missing form field `file`".to_string()))?;

    let (id, full_path) = state
        .storage
        .save(file_name.as_deref(), &data)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    tracing::info!(file = %full_path.display(), "Image uploaded to directory");

    // Attempt to describe the image via OpenAI Vision; failure should not block the upload response.
    let description = describe_image(&full_path).await.ok();

    let response = UploadResponse {
        id,
        path: full_path.display().to_string(),
        description,
    };

    Ok((StatusCode::CREATED, Json(response)))
}

async fn describe_image(path: &Path) -> Result<String, String> {
    let api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY env var not set; skipping description".to_string())?;

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| format!("Failed to read image for description: {e}"))?;

    let mime_hint = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        match ext.to_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "avif" => "image/avif",
            _ => "application/octet-stream",
        }
    } else {
        "application/octet-stream"
    };

    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    let data_url = format!("data:{mime_hint};base64,{b64}");

    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "Briefly describe this image." },
                { "type": "image_url", "image_url": { "url": data_url } }
            ]
        }],
        "max_tokens": 80
    });

    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Vision request failed: {e}"))?;

    let value: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse vision response: {e}"))?;

    value["choices"]
        .get(0)
        .and_then(|c| c["message"]["content"].as_str())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "Vision response missing description".to_string())
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/upload", post(upload_image))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        application::items::ItemService,
        infra::{repo::memory::InMemoryItemRepo, storage::FileStorage},
    };
    use axum::{
        body::{self, Body},
        http::{Request, StatusCode},
    };
    use std::{env, sync::Arc};
    use tempfile::tempdir;
    use tower::ServiceExt;

    #[tokio::test]
    async fn upload_saves_file() {
        let temp = tempdir().unwrap();
        env::set_current_dir(&temp).unwrap();

        let repo = Arc::new(InMemoryItemRepo::new_shared());
        let state = AppState {
            items: ItemService::new(repo),
            storage: FileStorage::new("uploads"),
        };

        let app = super::routes().with_state(state);

        let boundary = "XBOUNDARY";
        let form_body = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\n{data}\r\n--{b}--\r\n",
            b = boundary,
            data = "PNGDATA"
        );

        let request = Request::builder()
            .method("POST")
            .uri("/upload")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            )
            .body(Body::from(form_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        let body = body::to_bytes(response.into_body(), 1024).await.unwrap();
        let parsed: UploadResponse = serde_json::from_slice(&body).unwrap();

        let saved_path = Path::new(&parsed.path);
        assert!(saved_path.exists(), "file should have been written");
        assert_eq!(saved_path.parent().unwrap().file_name().unwrap(), "uploads");
        assert!(parsed.description.is_none(), "no description without OPENAI_API_KEY");
    }
}
