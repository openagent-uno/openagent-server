use anyhow::Result;
use rmcp::{ServerHandler, ServiceExt, transport::stdio};
use tracing_subscriber::{EnvFilter, fmt};

mod scaling;

#[derive(Clone, Default)]
struct ComputerControlServer;

impl ServerHandler for ComputerControlServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::new(
            "openagent-computer-control",
            env!("CARGO_PKG_VERSION"),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let service = ComputerControlServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
