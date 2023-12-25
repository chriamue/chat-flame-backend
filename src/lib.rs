//! # Chat Flame Backend Library
//!
//! `chat_flame_backend` is a library for building chat application backends.
//! It leverages the power of the Candle AI framework, with a focus on the Mistral model and other language models.
//!
//! This library provides the necessary modules to set up and run a backend server,
//! handle API requests, manage configuration, and interact with language models.

/// The `api` module contains the definitions and implementations of the API endpoints.
/// This includes routes for generating text, streaming responses, retrieving model information, etc.
pub mod api;

/// The `config` module manages the application's configuration.
/// It includes functionality for loading and parsing configuration files,
/// and provides access to configuration parameters throughout the application.
pub mod config;

/// The `llm` (Language Model) module contains the implementation and utilities related to language models.
/// This includes tokenization, text generation, model interfaces, and other language model-related functionality.
pub mod llm;

/// The `server` module is responsible for setting up and running the web server.
/// It includes the definition of routes, middleware, and other server-related configurations.
pub mod server;
