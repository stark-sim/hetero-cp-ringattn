#![allow(dead_code)]

//! NIXL block-direct transport, implementing the block-transport trait.
//!
//! Feature-gated behind nixl-backend (default off). Backed by the official
//! nixl-sys crate (bindgen FFI + safe Agent/XferDescList/XferRequest wrappers).
//! nixl-sys needs libclang at build time, so this module only compiles on
//! white/pearl (clang + libclang installed). With the stub-api feature,
//! libnixl_capi.so is dlopen'd at runtime, so it works against either the
//! pearl source build tree or the white conda wheel.
//!
//! Lifecycle mirrors SerializedBlockTransport: register a block -> exchange
//! agent metadata via the side channel -> submit_transfer posts an async NIXL
//! transfer -> poll_transfers drains completions and reports telemetry bytes
//! into the K10 wire-byte ledger.

#[cfg(feature = "nixl-backend")]
use crate::model::transport::block_transport::{
    BlockDesc, BlockHandle, KvBlockTransport, RemoteBlockDesc, TransferCompletion,
};
#[cfg(feature = "nixl-backend")]
use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, NixlError, OptArgs,
    RegistrationHandle, ThreadSync, XferDescList, XferOp, XferRequest,
};
#[cfg(feature = "nixl-backend")]
use std::collections::HashMap;
#[cfg(feature = "nixl-backend")]
use tch::Tensor;

/// Map a nixl_sys error to a String for the KvBlockTransport trait error type.
#[cfg(feature = "nixl-backend")]
fn map_err(e: NixlError) -> String {
    format!("nixl: {e:?}")
}

/// A VRAM memory region implementing the nixl-sys descriptor traits so a tch
/// tensor's device pointer can be registered with NIXL.
#[cfg(feature = "nixl-backend")]
#[derive(Debug)]
struct VramRegion {
    ptr: usize,
    len: usize,
    dev_id: u64,
}

// SAFETY: VramRegion is plain data (pointer + length); the device memory it
// refers to is owned by the caller and outlives the descriptor.
#[cfg(feature = "nixl-backend")]
unsafe impl Send for VramRegion {}
#[cfg(feature = "nixl-backend")]
unsafe impl Sync for VramRegion {}

#[cfg(feature = "nixl-backend")]
impl MemoryRegion for VramRegion {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn size(&self) -> usize {
        self.len
    }
}

#[cfg(feature = "nixl-backend")]
impl NixlDescriptor for VramRegion {
    fn mem_type(&self) -> MemType {
        // TEMP DRAM experiment: cross-vendor (CUDA<->ROCm) GPU-direct VRAM put
        // has no UCX remote protocol; host-memory (DRAM) transfer over tcp is
        // the fallback to validate first.
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        self.dev_id
    }
}

/// NIXL block transport: register device-memory blocks, exchange agent
/// metadata via the side channel, and move blocks with async NIXL transfers.
#[cfg(feature = "nixl-backend")]
pub struct NixlBlockTransport {
    agent_name: String,
    agent: Agent,
    /// The UCX backend instantiated at construction; opt_args reference it.
    backend: Backend,
    opt_args: OptArgs,
    next_block_id: u64,
    /// block id -> (advertised descriptor, registration handle).
    blocks: HashMap<u64, (BlockDesc, RegistrationHandle)>,
    /// (remote block id, local byte len, xfer request). The local byte
    /// len is carried so poll_transfers can fill the K10 wire-byte ledger
    /// even when NIXL telemetry is disabled (get_telemetry -> NoTelemetry).
    pending: Vec<(u64, u64, XferRequest)>,
    wire_sent: u64,
    wire_recv: u64,
}

// SAFETY: All NIXL handles (Agent/Backend/OptArgs/RegistrationHandle/
// XferRequest) are owned exclusively by this struct and accessed from the
// single worker thread that owns it. This mirrors nixl-sys's own unsafe
// Send/Sync impls for Backend and XferRequest.
#[cfg(feature = "nixl-backend")]
unsafe impl Send for NixlBlockTransport {}

#[cfg(feature = "nixl-backend")]
impl NixlBlockTransport {
    /// The agent name this transport registered under.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Create a NIXL agent with a unique name, and instantiate the UCX
    /// backend so VRAM registration succeeds. The official example shows this
    /// is required: a bare agent has "no available backends for VRAM_SEG".
    pub fn new(name: &str) -> Result<Self, String> {
        // Cross-machine transfers need the listen thread so this agent's UCX
        // backend accepts the peer's incoming connection; the default
        // enable_listen_thread=false makes the agent client-only, which fails
        // load_remote_md with "UCX endpoint create failed: Connection refused".
        // capture_telemetry=true makes get_telemetry() available without the
        // NIXL_TELEMETRY_ENABLE env var.
        let cfg = AgentConfig {
            enable_prog_thread: true,
            enable_listen_thread: true,
            // 0 lets NIXL/OS pick an ephemeral port so repeated probes (and
            // future co-located agents) never collide on a fixed listen_port.
            listen_port: 0,
            thread_sync: ThreadSync::None,
            num_workers: 1,
            pthr_delay_us: 0,
            lthr_delay_us: 100_000,
            capture_telemetry: true,
        };
        let agent = Agent::new_configured(name, &cfg).map_err(map_err)?;

        // Discover plugins and confirm UCX is available.
        let plugins = agent.get_available_plugins().map_err(map_err)?;
        let mut found_ucx = false;
        for plugin in plugins.iter() {
            if plugin.map_err(map_err)? == "UCX" {
                found_ucx = true;
            }
        }
        if !found_ucx {
            return Err("NIXL UCX backend not available".to_string());
        }

        // Instantiate the UCX backend with default plugin params.
        let (_mems, params) = agent.get_plugin_params("UCX").map_err(map_err)?;
        let backend = agent.create_backend("UCX", &params).map_err(map_err)?;

        // opt_args carrying the backend; register_mem / transfers reference it.
        let mut opt_args = OptArgs::new().map_err(map_err)?;
        opt_args.add_backend(&backend).map_err(map_err)?;

        Ok(Self {
            agent_name: name.to_string(),
            agent,
            backend,
            opt_args,
            next_block_id: 0,
            blocks: HashMap::new(),
            pending: Vec::new(),
            wire_sent: 0,
            wire_recv: 0,
        })
    }
}

#[cfg(feature = "nixl-backend")]
impl KvBlockTransport for NixlBlockTransport {
    fn register_block(&mut self, tensor: &Tensor) -> Result<BlockHandle, String> {
        let id = self.next_block_id;
        self.next_block_id += 1;
        let elem_bytes: usize = match tensor.kind() {
            tch::Kind::Float | tch::Kind::Int64 => 4,
            tch::Kind::Half | tch::Kind::BFloat16 => 2,
            tch::Kind::Double => 8,
            _ => 4,
        };
        let desc = BlockDesc {
            addr: tensor.data_ptr() as u64,
            len: (tensor.numel() * elem_bytes) as u64,
            dev_id: 0,
            meta: serde_json::to_vec(&tensor.size())
                .map_err(|e| format!("serialize shape meta failed: {e}"))?,
        };

        let region = VramRegion {
            ptr: desc.addr as usize,
            len: desc.len as usize,
            dev_id: desc.dev_id,
        };
        let handle = self
            .agent
            .register_memory(&region, Some(&self.opt_args))
            .map_err(map_err)?;
        self.blocks.insert(id, (desc.clone(), handle));
        Ok(BlockHandle { id, desc })
    }

    fn deregister_block(&mut self, handle: &BlockHandle) -> Result<(), String> {
        // RegistrationHandle deregisters on drop (nixl-sys Drop impl).
        self.blocks
            .remove(&handle.id)
            .ok_or_else(|| format!("block {} not registered", handle.id))?;
        Ok(())
    }

    fn local_metadata(&self) -> Result<Vec<u8>, String> {
        self.agent.get_local_md().map_err(map_err)
    }

    fn load_remote_metadata(&mut self, blob: &[u8]) -> Result<String, String> {
        self.agent.load_remote_md(blob).map_err(map_err)
    }

    fn submit_transfer(
        &mut self,
        local: &BlockHandle,
        remote: &RemoteBlockDesc,
    ) -> Result<(), String> {
        let local_desc = self
            .blocks
            .get(&local.id)
            .map(|(d, _)| d.clone())
            .ok_or_else(|| format!("local block {} not registered", local.id))?;

        let mut local_dlist = XferDescList::new(MemType::Dram).map_err(map_err)?;
        local_dlist.add_desc(
            local_desc.addr as usize,
            local_desc.len as usize,
            local_desc.dev_id,
        );
        let mut remote_dlist = XferDescList::new(MemType::Dram).map_err(map_err)?;
        remote_dlist.add_desc(
            remote.desc.addr as usize,
            remote.desc.len as usize,
            remote.desc.dev_id,
        );

        let req = self
            .agent
            .create_xfer_req(
                XferOp::Write,
                &local_dlist,
                &remote_dlist,
                &remote.agent,
                Some(&self.opt_args),
            )
            .map_err(map_err)?;
        let _in_progress = self
            .agent
            .post_xfer_req(&req, Some(&self.opt_args))
            .map_err(map_err)?;
        self.wire_sent += local_desc.len;
        self.pending.push((remote.block_id, local_desc.len, req));
        Ok(())
    }

    fn poll_transfers(&mut self) -> Result<Vec<TransferCompletion>, String> {
        let mut completions = Vec::new();
        let mut remaining = Vec::new();
        for (block_id, local_len, req) in self.pending.drain(..) {
            let status = self.agent.get_xfer_status(&req).map_err(map_err)?;
            if status.is_success() {
                // telemetry may be disabled (NoTelemetry); fall back to the
                // advertised block length so the K10 wire-byte ledger stays
                // accurate. total_bytes is the transfer's true byte count when
                // telemetry is on (NIXL_TELEMETRY_ENABLE + NIXL_TELEMETRY_DIR).
                let bytes = match req.get_telemetry() {
                    Ok(t) => t.total_bytes,
                    Err(_) => local_len,
                };
                self.wire_recv += bytes;
                completions.push(TransferCompletion { block_id, bytes });
            } else {
                remaining.push((block_id, local_len, req));
            }
        }
        self.pending = remaining;
        Ok(completions)
    }

    fn wire_bytes_sent(&self) -> u64 {
        self.wire_sent
    }

    fn wire_bytes_recv(&self) -> u64 {
        self.wire_recv
    }
}
