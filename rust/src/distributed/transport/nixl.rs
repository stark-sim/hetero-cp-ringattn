#![allow(dead_code)]

//! NIXL block-direct transport, implementing the block-transport trait.
//!
//! Feature-gated behind nixl-backend (default off). Hand-declares the stable
//! NIXL C API (nixl_capi_* exported by libnixl_capi.so), so it type-checks
//! without libclang (unlike the bindgen-based nixl-sys crate) and links only
//! when the feature is enabled on a CUDA/ROCm host with libnixl_capi.so.
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
use std::collections::HashMap;
#[cfg(feature = "nixl-backend")]
use std::ffi::c_void;
#[cfg(feature = "nixl-backend")]
use std::os::raw::{c_char, c_int};
#[cfg(feature = "nixl-backend")]
use tch::Tensor;

// ===== Raw C ABI (mirrors wrapper.h, stable exported by libnixl_capi.so) =====

#[cfg(feature = "nixl-backend")]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NixlStatus {
    Success = 0,
    ErrorInvalidParam = -1,
    ErrorBackend = -2,
    ErrorInvalidState = -3,
    ErrorException = -4,
    InProg = 1,
    ErrorNoTelemetry = -5,
}

#[cfg(feature = "nixl-backend")]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NixlMemType {
    Dram = 0,
    Vram = 1,
    Block = 2,
    Object = 3,
    File = 4,
    Unknown = 5,
}

#[cfg(feature = "nixl-backend")]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NixlXferOp {
    Read = 0,
    Write = 1,
}

/// Telemetry struct layout (fixed 5x u64), from wrapper.h.
#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlXferTelemetry {
    start_time_us: u64,
    post_duration_us: u64,
    xfer_duration_us: u64,
    total_bytes: u64,
    desc_count: u64,
}

#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlAgentOpaque {
    _private: [u8; 0],
}
#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlRegDlistOpaque {
    _private: [u8; 0],
}
#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlXferDlistOpaque {
    _private: [u8; 0],
}
#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlXferReqOpaque {
    _private: [u8; 0],
}
#[cfg(feature = "nixl-backend")]
#[repr(C)]
struct NixlOptArgsOpaque {
    _private: [u8; 0],
}

#[cfg(feature = "nixl-backend")]
type NixlAgent = *mut NixlAgentOpaque;
#[cfg(feature = "nixl-backend")]
type NixlRegDlist = *mut NixlRegDlistOpaque;
#[cfg(feature = "nixl-backend")]
type NixlXferDlist = *mut NixlXferDlistOpaque;
#[cfg(feature = "nixl-backend")]
type NixlXferReq = *mut NixlXferReqOpaque;
#[cfg(feature = "nixl-backend")]
type NixlOptArgs = *mut NixlOptArgsOpaque;

#[cfg(feature = "nixl-backend")]
#[link(name = "nixl_capi")]
extern "C" {
    fn nixl_capi_create_agent(name: *const c_char, agent: *mut NixlAgent) -> c_int;
    fn nixl_capi_destroy_agent(agent: NixlAgent) -> c_int;

    fn nixl_capi_create_reg_dlist(mem_type: c_int, dlist: *mut NixlRegDlist) -> c_int;
    fn nixl_capi_destroy_reg_dlist(dlist: NixlRegDlist) -> c_int;
    fn nixl_capi_reg_dlist_add_desc(
        dlist: NixlRegDlist,
        addr: usize,
        len: usize,
        dev_id: u64,
        metadata: *const c_void,
        metadata_len: usize,
    ) -> c_int;
    fn nixl_capi_register_mem(
        agent: NixlAgent,
        dlist: NixlRegDlist,
        opt_args: NixlOptArgs,
    ) -> c_int;
    fn nixl_capi_deregister_mem(
        agent: NixlAgent,
        dlist: NixlRegDlist,
        opt_args: NixlOptArgs,
    ) -> c_int;

    fn nixl_capi_get_local_md(agent: NixlAgent, data: *mut *mut c_void, len: *mut usize) -> c_int;
    fn nixl_capi_load_remote_md(
        agent: NixlAgent,
        data: *const c_void,
        len: usize,
        agent_name: *mut *mut c_char,
    ) -> c_int;
    fn nixl_capi_agent_make_connection(
        agent: NixlAgent,
        remote_agent: *const c_char,
        opt_args: NixlOptArgs,
    ) -> c_int;

    fn nixl_capi_create_xfer_dlist(mem_type: c_int, dlist: *mut NixlXferDlist) -> c_int;
    fn nixl_capi_destroy_xfer_dlist(dlist: NixlXferDlist) -> c_int;
    fn nixl_capi_xfer_dlist_add_desc(
        dlist: NixlXferDlist,
        addr: usize,
        len: usize,
        dev_id: u64,
    ) -> c_int;

    fn nixl_capi_create_xfer_req(
        agent: NixlAgent,
        operation: c_int,
        local_descs: NixlXferDlist,
        remote_descs: NixlXferDlist,
        remote_agent: *const c_char,
        req_hndl: *mut NixlXferReq,
        opt_args: NixlOptArgs,
    ) -> c_int;
    fn nixl_capi_post_xfer_req(agent: NixlAgent, req: NixlXferReq, opt_args: NixlOptArgs) -> c_int;
    fn nixl_capi_get_xfer_status(agent: NixlAgent, req: NixlXferReq) -> c_int;
    fn nixl_capi_release_xfer_req(agent: NixlAgent, req: NixlXferReq) -> c_int;
    fn nixl_capi_destroy_xfer_req(req: NixlXferReq) -> c_int;
    fn nixl_capi_get_xfer_telemetry(
        agent: NixlAgent,
        req: NixlXferReq,
        telemetry: *mut NixlXferTelemetry,
    ) -> c_int;

    fn nixl_capi_create_opt_args(args: *mut NixlOptArgs) -> c_int;
    fn nixl_capi_destroy_opt_args(args: NixlOptArgs) -> c_int;
}

#[cfg(feature = "nixl-backend")]
fn status_ok(status: c_int, what: &str) -> Result<(), String> {
    if status == NixlStatus::Success as i32 {
        Ok(())
    } else {
        Err(format!("{what} failed with status {status}"))
    }
}

/// NIXL block transport: register device-memory blocks, exchange agent
/// metadata via the side channel, and move blocks with async NIXL transfers.
#[cfg(feature = "nixl-backend")]
pub struct NixlBlockTransport {
    agent_name: String,
    agent: NixlAgent,
    next_block_id: u64,
    blocks: HashMap<u64, BlockDesc>,
    pending: Vec<(u64, NixlXferReq)>,
    wire_sent: u64,
    wire_recv: u64,
}

#[cfg(feature = "nixl-backend")]
impl NixlBlockTransport {
    /// Create a NIXL agent with a unique name.
    pub fn new(name: &str) -> Result<Self, String> {
        let c_name =
            std::ffi::CString::new(name).map_err(|e| format!("invalid agent name: {e}"))?;
        let mut agent: NixlAgent = std::ptr::null_mut();
        let status = unsafe { nixl_capi_create_agent(c_name.as_ptr(), &mut agent) };
        status_ok(status, "nixl_capi_create_agent")?;
        if agent.is_null() {
            return Err("nixl_capi_create_agent returned null agent".to_string());
        }
        Ok(Self {
            agent_name: name.to_string(),
            agent,
            next_block_id: 0,
            blocks: HashMap::new(),
            pending: Vec::new(),
            wire_sent: 0,
            wire_recv: 0,
        })
    }

    fn register_block_desc(&self, desc: &BlockDesc) -> Result<NixlRegDlist, String> {
        let mut dlist: NixlRegDlist = std::ptr::null_mut();
        let status = unsafe { nixl_capi_create_reg_dlist(NixlMemType::Vram as i32, &mut dlist) };
        status_ok(status, "nixl_capi_create_reg_dlist")?;
        let status = unsafe {
            nixl_capi_reg_dlist_add_desc(
                dlist,
                desc.addr as usize,
                desc.len as usize,
                desc.dev_id,
                desc.meta.as_ptr() as *const c_void,
                desc.meta.len(),
            )
        };
        if status != NixlStatus::Success as i32 {
            unsafe { nixl_capi_destroy_reg_dlist(dlist) };
            return Err(format!("nixl_capi_reg_dlist_add_desc failed: {status}"));
        }
        let status = unsafe { nixl_capi_register_mem(self.agent, dlist, std::ptr::null_mut()) };
        if status != NixlStatus::Success as i32 {
            unsafe { nixl_capi_destroy_reg_dlist(dlist) };
            return Err(format!("nixl_capi_register_mem failed: {status}"));
        }
        Ok(dlist)
    }
}

// SAFETY: NIXL C API handles are owned exclusively by this struct and are
// not shared across threads; the trait's Send bound is satisfied by the fact
// that all NIXL calls happen on the worker thread that owns the transport.
// This mirrors NIXL's own rust bindings, which wrap the same raw handles in
// NonNull (itself Send).
#[cfg(feature = "nixl-backend")]
unsafe impl Send for NixlBlockTransport {}

#[cfg(feature = "nixl-backend")]
impl Drop for NixlBlockTransport {
    fn drop(&mut self) {
        unsafe {
            let _ = nixl_capi_destroy_agent(self.agent);
        }
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
        let _dlist = self.register_block_desc(&desc)?;
        self.blocks.insert(id, desc.clone());
        Ok(BlockHandle { id, desc })
    }

    fn deregister_block(&mut self, handle: &BlockHandle) -> Result<(), String> {
        let desc = self
            .blocks
            .remove(&handle.id)
            .ok_or_else(|| format!("block {} not registered", handle.id))?;
        let mut dlist: NixlRegDlist = std::ptr::null_mut();
        let status = unsafe { nixl_capi_create_reg_dlist(NixlMemType::Vram as i32, &mut dlist) };
        status_ok(status, "nixl_capi_create_reg_dlist")?;
        unsafe {
            nixl_capi_reg_dlist_add_desc(
                dlist,
                desc.addr as usize,
                desc.len as usize,
                desc.dev_id,
                desc.meta.as_ptr() as *const c_void,
                desc.meta.len(),
            )
        };
        let status = unsafe { nixl_capi_deregister_mem(self.agent, dlist, std::ptr::null_mut()) };
        unsafe { nixl_capi_destroy_reg_dlist(dlist) };
        status_ok(status, "nixl_capi_deregister_mem")
    }

    fn local_metadata(&self) -> Result<Vec<u8>, String> {
        let mut data: *mut c_void = std::ptr::null_mut();
        let mut len: usize = 0;
        let status = unsafe { nixl_capi_get_local_md(self.agent, &mut data, &mut len) };
        status_ok(status, "nixl_capi_get_local_md")?;
        if data.is_null() || len == 0 {
            return Err("nixl_capi_get_local_md returned empty metadata".to_string());
        }
        Ok(unsafe { std::slice::from_raw_parts(data as *const u8, len) }.to_vec())
    }

    fn load_remote_metadata(&mut self, blob: &[u8]) -> Result<String, String> {
        let mut agent_name: *mut c_char = std::ptr::null_mut();
        let status = unsafe {
            nixl_capi_load_remote_md(
                self.agent,
                blob.as_ptr() as *const c_void,
                blob.len(),
                &mut agent_name,
            )
        };
        status_ok(status, "nixl_capi_load_remote_md")?;
        if agent_name.is_null() {
            return Err("nixl_capi_load_remote_md returned null agent name".to_string());
        }
        Ok(unsafe { std::ffi::CStr::from_ptr(agent_name) }
            .to_string_lossy()
            .into_owned())
    }

    fn submit_transfer(
        &mut self,
        local: &BlockHandle,
        remote: &RemoteBlockDesc,
    ) -> Result<(), String> {
        let local_desc = self
            .blocks
            .get(&local.id)
            .ok_or_else(|| format!("local block {} not registered", local.id))?;

        let mut local_dlist: NixlXferDlist = std::ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_xfer_dlist(NixlMemType::Vram as i32, &mut local_dlist) };
        status_ok(status, "nixl_capi_create_xfer_dlist")?;
        unsafe {
            nixl_capi_xfer_dlist_add_desc(
                local_dlist,
                local_desc.addr as usize,
                local_desc.len as usize,
                local_desc.dev_id,
            )
        };

        let mut remote_dlist: NixlXferDlist = std::ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_xfer_dlist(NixlMemType::Vram as i32, &mut remote_dlist) };
        status_ok(status, "nixl_capi_create_xfer_dlist")?;
        unsafe {
            nixl_capi_xfer_dlist_add_desc(
                remote_dlist,
                remote.desc.addr as usize,
                remote.desc.len as usize,
                remote.desc.dev_id,
            )
        };

        let remote_agent = std::ffi::CString::new(remote.agent.as_str())
            .map_err(|e| format!("invalid remote agent name: {e}"))?;

        let mut req: NixlXferReq = std::ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_xfer_req(
                self.agent,
                NixlXferOp::Write as i32,
                local_dlist,
                remote_dlist,
                remote_agent.as_ptr(),
                &mut req,
                std::ptr::null_mut(),
            )
        };
        unsafe {
            nixl_capi_destroy_xfer_dlist(local_dlist);
            nixl_capi_destroy_xfer_dlist(remote_dlist);
        }
        status_ok(status, "nixl_capi_create_xfer_req")?;

        let status = unsafe { nixl_capi_post_xfer_req(self.agent, req, std::ptr::null_mut()) };
        if status != NixlStatus::Success as i32 && status != NixlStatus::InProg as i32 {
            unsafe { nixl_capi_destroy_xfer_req(req) };
            return Err(format!("nixl_capi_post_xfer_req failed: {status}"));
        }
        self.wire_sent += local_desc.len;
        self.pending.push((remote.block_id, req));
        Ok(())
    }

    fn poll_transfers(&mut self) -> Result<Vec<TransferCompletion>, String> {
        let mut completions = Vec::new();
        let mut remaining = Vec::new();
        for (block_id, req) in self.pending.drain(..) {
            let status = unsafe { nixl_capi_get_xfer_status(self.agent, req) };
            if status == NixlStatus::Success as i32 {
                let mut telemetry = NixlXferTelemetry {
                    start_time_us: 0,
                    post_duration_us: 0,
                    xfer_duration_us: 0,
                    total_bytes: 0,
                    desc_count: 0,
                };
                let _ = unsafe { nixl_capi_get_xfer_telemetry(self.agent, req, &mut telemetry) };
                self.wire_recv += telemetry.total_bytes;
                completions.push(TransferCompletion {
                    block_id,
                    bytes: telemetry.total_bytes,
                });
                unsafe {
                    nixl_capi_release_xfer_req(self.agent, req);
                    nixl_capi_destroy_xfer_req(req);
                }
            } else if status == NixlStatus::InProg as i32 {
                remaining.push((block_id, req));
            } else {
                unsafe { nixl_capi_destroy_xfer_req(req) };
                return Err(format!("nixl_capi_get_xfer_status failed: {status}"));
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
