use crate::cli::CliArgs;
use crate::error::RingError;
use crate::protocol;

pub fn run_remote_p2p(args: &CliArgs) -> Result<protocol::RemoteP2pReport, RingError> {
    match args.remote_p2p_role.as_deref() {
        Some("server") => {
            let bind_addr = args.bind_addr.as_deref().unwrap_or("0.0.0.0:29172");
            Ok(protocol::run_remote_p2p_server(bind_addr)?)
        }
        Some("client") => {
            let connect_addr = args.connect_addr.as_deref().ok_or_else(|| {
                RingError::InvalidCli("--connect is required for remote client".to_string())
            })?;
            Ok(protocol::run_remote_p2p_client(connect_addr)?)
        }
        Some(role) => Err(RingError::InvalidCli(format!(
            "unsupported --remote-p2p-role {role}; expected server, client, or cp-node"
        ))),
        None => Err(RingError::InvalidCli(
            "--remote-p2p-role is required for remote mode".to_string(),
        )),
    }
}

pub fn run_remote_cp_node(args: &CliArgs) -> Result<protocol::RemoteCpNodeReport, RingError> {
    let node_index = args
        .node_index
        .ok_or_else(|| RingError::InvalidCli("--node-index is required for cp-node".to_string()))?;
    let bind_addr = args
        .bind_addr
        .as_deref()
        .ok_or_else(|| RingError::InvalidCli("--bind is required for cp-node".to_string()))?;
    let connect_addr = args
        .connect_addr
        .as_deref()
        .ok_or_else(|| RingError::InvalidCli("--connect is required for cp-node".to_string()))?;
    Ok(protocol::run_remote_cp_node(
        node_index,
        bind_addr,
        connect_addr,
    )?)
}
