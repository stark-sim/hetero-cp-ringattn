pub mod message;
pub mod transport;
pub mod framing;
pub mod node;

pub use message::ProtocolError;
pub(crate) use message::{
    RingAttnMessage, RingAttnMessageKind, PayloadKind,
    BlockMetadata, TensorMetadata, FLOAT32_BYTES, SCHEMA_VERSION,
};
pub use node::{
    CpPayloadBlock, CpRingNodeSmokeReport, ProtocolSmokeReport,
    RemoteCpNodeReport, RemoteP2pReport, run_cp_ring_node_smoke,
    run_protocol_smoke, run_remote_cp_node, run_remote_p2p_client,
    run_remote_p2p_server,
};
pub(crate) use node::{
    DomainModelState, LayerActivationState, OnlineSoftmaxAccumulator,
};
