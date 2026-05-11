#![allow(dead_code)]

#[cfg(feature = "tch-backend")]
use super::block::KvBlock;
use super::r#trait::KvTransport;

/// In-memory transport for unit testing distributed attention logic.
#[cfg(feature = "tch-backend")]
pub struct MockKvTransport {
    queue: std::collections::VecDeque<KvBlock>,
}

#[cfg(feature = "tch-backend")]
impl MockKvTransport {
    pub fn new() -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
        }
    }

    pub fn push(&mut self, block: KvBlock) {
        self.queue.push_back(block);
    }
}

#[cfg(feature = "tch-backend")]
impl KvTransport for MockKvTransport {
    fn send_kv_block(&mut self, _block: &KvBlock) -> Result<(), String> {
        Ok(())
    }

    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        Ok(self.queue.pop_front())
    }
}

/// 【双向内存传输通道】用于单进程测试中模拟两个分布式 worker 交换 KV block。
/// 
/// 在真实的分布式环境里，domain0 和 domain1 运行在两台不同的机器上，通过网络互相发送 KV。
/// 在单进程测试里，我们用这个结构模拟网络：
/// - domain0 发送的 KV block 会进入 domain1 的接收队列
/// - domain1 发送的 KV block 会进入 domain0 的接收队列
#[cfg(feature = "tch-backend")]
#[derive(Clone)]
pub struct LinkedMockKvTransport {
    // 【peer_inbox：对方的收件箱】
    // 当我们调用 send 时，数据会被推入这个队列。
    // 这个队列实际上是对端（peer）的 self_inbox，所以对方调用 recv 时就能读到。
    // 
    // 用 Arc<Mutex<VecDeque>> 的原因是：
    // - VecDeque：双端队列，支持从尾部 push、从头部 pop（先进先出）。
    // - Mutex：多线程锁，保证同一时间只有一个线程能读写队列。
    // - Arc：原子引用计数，让多个 transport 实例共享同一个队列（不拷贝数据）。
    peer_inbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<KvBlock>>>,

    // 【self_inbox：自己的收件箱】
    // 当我们调用 recv 时，数据会从这个队列弹出。
    // 这个队列由对端在 send 时写入。
    self_inbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<KvBlock>>>,
}

#[cfg(feature = "tch-backend")]
impl LinkedMockKvTransport {
    /// 【创建一对互通的传输通道】返回 (t0, t1)。
    /// 
    /// 核心设计：交叉共享队列，让 t0 的发送等于 t1 的接收，反之亦然。
    /// 
    /// 具体做法：
    /// - 创建两个空队列 q0 和 q1。
    /// - t0.peer_inbox = q1（t0 发送 → 写入 q1）
    /// - t0.self_inbox = q0（t0 接收 → 从 q0 读取）
    /// - t1.peer_inbox = q0（t1 发送 → 写入 q0）
    /// - t1.self_inbox = q1（t1 接收 → 从 q1 读取）
    /// 
    /// 结果：
    /// - t0.send() 的数据会被 t1.recv() 读到
    /// - t1.send() 的数据会被 t0.recv() 读到
    pub fn create_pair() -> (Self, Self) {
        let q0 = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        let q1 = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        (
            Self { peer_inbox: q1.clone(), self_inbox: q0.clone() },
            Self { peer_inbox: q0.clone(), self_inbox: q1.clone() },
        )
    }
}

#[cfg(feature = "tch-backend")]
impl KvTransport for LinkedMockKvTransport {
    /// 【发送 KV block】把 block 的副本放入对方的收件箱（peer_inbox）。
    /// 
    // shallow_clone() 不拷贝底层浮点数据，只增加引用计数，所以很高效。
    fn send_kv_block(&mut self, block: &KvBlock) -> Result<(), String> {
        let cloned = KvBlock {
            layer_idx: block.layer_idx,
            global_seq_start: block.global_seq_start,
            global_seq_end: block.global_seq_end,
            k: block.k.shallow_clone(),
            v: block.v.shallow_clone(),
        };
        self.peer_inbox.lock().unwrap().push_back(cloned);
        Ok(())
    }

    /// 【接收 KV block】从自己的收件箱（self_inbox）头部取出一个 block。
    /// 
    /// 如果队列为空，返回 Ok(None)，表示暂时没有数据。
    fn recv_kv_block(&mut self) -> Result<Option<KvBlock>, String> {
        Ok(self.self_inbox.lock().unwrap().pop_front())
    }
}
