// api/mod.rs
//! The `api` module provides the necessary components to build the API for the chat-flame-backend.
//! It includes definitions for models, routes, and OpenAPI documentation.
//!
//! This module is responsible for handling all the HTTP requests and responses,
//! structuring the JSON data, and providing the necessary endpoints for the application.

pub mod model; // Models used in the API for request and response data structures.
pub mod openapi; // OpenAPI documentation and specifications.
pub mod routes; // Definitions of all the API routes and their handlers.
