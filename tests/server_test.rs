use axum_test::TestServer;
use chat_flame_backend::config::Config;
use chat_flame_backend::server::server;

#[ignore = "ignore until mocked"]
#[tokio::test]
async fn test_generate_handler() {
    let config = Config::default();
    let app = server(config);

    let server = TestServer::new(app).unwrap();
    let response = server
        .post("/")
        .json(&serde_json::json!({
            "inputs": "write hello world in rust",
            "parameters": {
                "temperature": 0.9,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "top_n_tokens": 64,
                "max_new_tokens": 50,
                "stop": ["</s>"]
            }
        }))
        .await;

    assert_eq!(response.status_code(), 200);
}

#[ignore = "ignore until mocked"]
#[tokio::test]
async fn test_generate_text_handler() {
    let config = Config::default();
    let app = server(config);

    let server = TestServer::new(app).unwrap();
    let response = server
        .post("/generate")
        .json(&serde_json::json!({
            "inputs": "write hello world in rust",
            "parameters": {
                "temperature": 0.9,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "top_n_tokens": 64,
                "max_new_tokens": 50,
                "stop": ["</s>"]
            }
        }))
        .await;

    assert_eq!(response.status_code(), 200);
}

//#[ignore = "ignore until mocked"]
#[tokio::test]
async fn test_generate_text_model_handler() {
    let config = Config::default();
    let app = server(config);

    let server = TestServer::new(app).unwrap();
    let response = server
        .post("/model/phi-v2")
        .json(&serde_json::json!({
            "inputs": "write hello world in rust",
            "parameters": {
                "temperature": 0.9,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "top_n_tokens": 64,
                "max_new_tokens": 50,
                "stop": ["</s>"]
            }
        }))
        .await;

    assert_eq!(response.status_code(), 200);
}

#[tokio::test]
async fn test_get_health_handler() {
    let config = Config::default();
    let app = server(config);

    let server = TestServer::new(app).unwrap();
    let response = server.get("/health").await;

    assert_eq!(response.status_code(), 200);
}

#[tokio::test]
async fn test_get_info_handler() {
    let config = Config::default();
    let app = server(config);

    let server = TestServer::new(app).unwrap();
    let response = server.get("/info").await;

    assert_eq!(response.status_code(), 200);
}
