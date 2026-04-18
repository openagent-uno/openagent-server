mod capture;
mod input;
mod keys;
mod record;
mod scaling;
mod tool;

use anyhow::Result;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let server = tool::ComputerControlServer::new();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
