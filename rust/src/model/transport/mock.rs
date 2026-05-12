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
    /// 【创建 N-domain ring 传输通道】返回 N 个 transport，组成一个 ring。
    ///
    /// domain i 的 send 会进入 domain (i+1)%N 的 recv：
    /// - t[i].peer_inbox = q[(i+1)%N]（t[i] 发送 → 写入 q[(i+1)%N]）
    /// - t[i].self_inbox = q[i]（t[i] 接收 → 从 q[i] 读取）
    ///
    /// 在 ring attention 中：
    /// - Round 0: domain i 发送本地 KV → domain (i+1)%N 收到
    /// - Round 1: domain i 转发收到的 KV → domain (i+1)%N 收到
    /// - 经过 N-1 轮后，每个 domain 都收到了所有其他 domain 的 KV。
    pub fn create_ring(n: usize) -> Vec<Self> {
        assert!(n >= 2, "ring must have at least 2 domains");
        let queues: Vec<_> = (0..n)
            .map(|_| std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())))
            .collect();
        (0..n)
            .map(|i| Self {
                peer_inbox: queues[(i + 1) % n].clone(),
                self_inbox: queues[i].clone(),
            })
            .collect()
    }

    /// 【创建一对互通的传输通道】返回 (t0, t1)。
    /// 等价于 create_ring(2)。
    pub fn create_pair() -> (Self, Self) {
        let ring = Self::create_ring(2);
        (ring[0].clone(), ring[1].clone())
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
