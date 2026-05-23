//! 【Continuous Batching 调度器 — Coordinator 侧】
//!
//! 负责管理请求生命周期和迭代调度：
//! - `pending` 队列：等待 prefill 的请求
//! - `active` 池：已完成 prefill，正在 decode batch 中的请求
//! - 每次 iteration 可以：prefill 新请求（如果 batch 未满）+ decode 所有 active 请求
//!
//! Phase 1 简化策略（Prefill-Then-Decode）：
//! - 不支持 mixed batching（prefill 和 decode 不混合在同一个 worker forward 中）
//! - 新请求串行 prefill，prefill 完成后加入 active decode pool
//! - 所有 active 请求每 iteration 一起发送 DecodeBatch
//! - 完成的请求立即移出，不影响其他请求

use std::collections::{HashMap, VecDeque};
use crate::api::types::{InferenceJob, InferenceResult};

/// 一个处于 decode batch 中的活跃请求。
pub struct ActiveRequest {
    pub request_id: u64,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,

    /// Tokenized prompt IDs。
    pub prompt_ids: Vec<i64>,
    pub prompt_tokens: usize,

    /// Chunk boundaries for distributed prefill。
    /// `chunk_boundaries[i]` = start index of domain i's chunk。
    /// Length = num_domains + 1 (last element = prompt_tokens)。
    pub chunk_boundaries: Vec<usize>,

    /// 已生成的 token IDs。
    pub generated_ids: Vec<u32>,
    /// 下一步要发送给 worker 的 token（上一步采样结果）。
    pub next_token: i64,
    /// 完成原因（None = 还在生成）。
    pub finish_reason: Option<String>,

    /// 结果回传通道。
    pub result_tx: tokio::sync::oneshot::Sender<InferenceResult>,
}

/// Batch 调度器。
///
/// 调度策略（Phase 1）：
/// - 固定 `max_batch_size`
/// - 新请求从 `pending` 移动到 `active` 需要经过 prefill（串行）
/// - 每次 iteration 对所有 active 请求发送 DecodeBatch
/// - 请求完成后立即从 active 移出
pub struct BatchScheduler {
    /// 等待 prefill 的请求队列。
    pending: VecDeque<InferenceJob>,
    /// 当前 decode batch 中的活跃请求。
    active: HashMap<u64, ActiveRequest>,
    /// Batch 最大容量。
    max_batch_size: usize,
}

impl BatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            active: HashMap::new(),
            max_batch_size: max_batch_size.max(1),
        }
    }

    /// 将新请求加入 pending 队列。
    pub fn enqueue(&mut self, job: InferenceJob) {
        self.pending.push_back(job);
    }

    /// 是否还可以接纳新请求到 active batch。
    pub fn can_admit(&self) -> bool {
        self.active.len() < self.max_batch_size
    }

    /// 尝试从 pending 队列中取出一个请求（如果 batch 未满）。
    pub fn try_dequeue_pending(&mut self) -> Option<InferenceJob> {
        if self.can_admit() {
            self.pending.pop_front()
        } else {
            None
        }
    }

    /// 将一个已完成 prefill 的请求加入 active pool。
    pub fn add_active(&mut self, req: ActiveRequest) {
        self.active.insert(req.request_id, req);
    }

    /// Active decode batch 是否为空。
    pub fn active_is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Pending 队列是否为空。
    pub fn pending_is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Active batch 中的请求数量。
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// 获取所有 active 请求的 ID 列表。
    pub fn active_request_ids(&self) -> Vec<u64> {
        self.active.keys().copied().collect()
    }

    /// 获取指定 active 请求的可变引用。
    pub fn get_active_mut(&mut self, request_id: u64) -> Option<&mut ActiveRequest> {
        self.active.get_mut(&request_id)
    }

    /// 从 active pool 中移除一个请求（通常是因为它已完成）。
    pub fn remove_active(&mut self, request_id: u64) -> Option<ActiveRequest> {
        self.active.remove(&request_id)
    }

    /// 是否还有工作要做（active 或 pending 不为空）。
    pub fn has_work(&self) -> bool {
        !self.active_is_empty() || !self.pending_is_empty()
    }

    /// 只读访问 active 请求。
    pub fn active_requests(&self) -> &HashMap<u64, ActiveRequest> {
        &self.active
    }

    /// 可变访问 active 请求。
    pub fn active_requests_mut(&mut self) -> &mut HashMap<u64, ActiveRequest> {
        &mut self.active
    }

    /// Pending 队列长度。
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(request_id: u64) -> InferenceJob {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        InferenceJob {
            request_id,
            prompt: format!("prompt-{request_id}"),
            max_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
            tx,
        }
    }

    #[test]
    fn test_scheduler_enqueue_and_dequeue() {
        let mut scheduler = BatchScheduler::new(2);
        scheduler.enqueue(make_job(1));
        scheduler.enqueue(make_job(2));
        scheduler.enqueue(make_job(3));

        assert_eq!(scheduler.pending_count(), 3);
        assert!(scheduler.can_admit());

        let job1 = scheduler.try_dequeue_pending().unwrap();
        assert_eq!(job1.request_id, 1);
        assert_eq!(scheduler.pending_count(), 2);
    }

    #[test]
    fn test_scheduler_max_batch_size() {
        let mut scheduler = BatchScheduler::new(2);

        // Admit 2 requests into active
        scheduler.enqueue(make_job(1));
        scheduler.enqueue(make_job(2));
        scheduler.enqueue(make_job(3));

        let j1 = scheduler.try_dequeue_pending().unwrap();
        // Simulate prefill completion — add j1 to active immediately
        let (tx1, _rx1) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j1.request_id,
            prompt: j1.prompt,
            max_tokens: j1.max_tokens,
            temperature: j1.temperature,
            top_p: j1.top_p,
            prompt_ids: vec![1, 2, 3],
            prompt_tokens: 3,
            chunk_boundaries: vec![0, 3],
            generated_ids: vec![42],
            next_token: 42,
            finish_reason: None,
            result_tx: tx1,
        });

        let j2 = scheduler.try_dequeue_pending().unwrap();
        // Add j2 to active — now batch is full
        let (tx2, _rx2) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j2.request_id,
            prompt: "p2".to_string(),
            max_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
            prompt_ids: vec![4, 5],
            prompt_tokens: 2,
            chunk_boundaries: vec![0, 2],
            generated_ids: vec![99],
            next_token: 99,
            finish_reason: None,
            result_tx: tx2,
        });

        assert_eq!(scheduler.active_len(), 2);
        assert!(!scheduler.can_admit());
        assert!(scheduler.try_dequeue_pending().is_none()); // batch full

        // Remove one — now we can admit again
        scheduler.remove_active(j1.request_id);
        assert_eq!(scheduler.active_len(), 1);
        assert!(scheduler.can_admit());

        // Third job can now be dequeued
        let j3 = scheduler.try_dequeue_pending().unwrap();
        assert_eq!(j3.request_id, 3);
    }

    #[test]
    fn test_scheduler_has_work() {
        let mut scheduler = BatchScheduler::new(2);
        assert!(!scheduler.has_work());

        scheduler.enqueue(make_job(1));
        assert!(scheduler.has_work());

        let j = scheduler.try_dequeue_pending().unwrap();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j.request_id,
            prompt: j.prompt,
            max_tokens: j.max_tokens,
            temperature: j.temperature,
            top_p: j.top_p,
            prompt_ids: vec![1],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![10],
            next_token: 10,
            finish_reason: None,
            result_tx: tx,
        });
        assert!(scheduler.has_work());

        scheduler.remove_active(j.request_id);
        assert!(!scheduler.has_work());
    }

    #[test]
    fn test_scheduler_dynamic_batch_size() {
        let mut scheduler = BatchScheduler::new(2);

        // Enqueue 3 jobs
        scheduler.enqueue(make_job(1));
        scheduler.enqueue(make_job(2));
        scheduler.enqueue(make_job(3));

        // Admit 2 into active (batch full)
        let j1 = scheduler.try_dequeue_pending().unwrap();
        let (tx1, _rx1) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j1.request_id,
            prompt: j1.prompt,
            max_tokens: j1.max_tokens,
            temperature: j1.temperature,
            top_p: j1.top_p,
            prompt_ids: vec![1],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![10],
            next_token: 100,
            finish_reason: None,
            result_tx: tx1,
        });

        let j2 = scheduler.try_dequeue_pending().unwrap();
        let (tx2, _rx2) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j2.request_id,
            prompt: j2.prompt,
            max_tokens: j2.max_tokens,
            temperature: j2.temperature,
            top_p: j2.top_p,
            prompt_ids: vec![2],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![20],
            next_token: 200,
            finish_reason: None,
            result_tx: tx2,
        });

        assert_eq!(scheduler.active_len(), 2);
        assert!(!scheduler.can_admit());
        assert!(scheduler.try_dequeue_pending().is_none()); // batch full

        // Simulate iteration: collect tokens from all active requests
        let tokens: Vec<(u64, i64)> = scheduler.active_requests()
            .values()
            .map(|req| (req.request_id, req.next_token))
            .collect();
        assert_eq!(tokens.len(), 2);
        assert!(tokens.iter().any(|(id, _)| *id == 1));
        assert!(tokens.iter().any(|(id, _)| *id == 2));

        // Request 1 completes — remove it
        scheduler.remove_active(1);
        assert_eq!(scheduler.active_len(), 1);
        assert!(scheduler.can_admit());

        // Now job 3 can be admitted
        let j3 = scheduler.try_dequeue_pending().unwrap();
        assert_eq!(j3.request_id, 3);
        let (tx3, _rx3) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j3.request_id,
            prompt: j3.prompt,
            max_tokens: j3.max_tokens,
            temperature: j3.temperature,
            top_p: j3.top_p,
            prompt_ids: vec![3],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![30],
            next_token: 300,
            finish_reason: None,
            result_tx: tx3,
        });

        assert_eq!(scheduler.active_len(), 2);
        assert!(!scheduler.can_admit());

        // Verify active request IDs
        let ids = scheduler.active_request_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_scheduler_active_requests_mut() {
        let mut scheduler = BatchScheduler::new(2);
        scheduler.enqueue(make_job(1));

        let j = scheduler.try_dequeue_pending().unwrap();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        scheduler.add_active(ActiveRequest {
            request_id: j.request_id,
            prompt: j.prompt,
            max_tokens: j.max_tokens,
            temperature: j.temperature,
            top_p: j.top_p,
            prompt_ids: vec![1],
            prompt_tokens: 1,
            chunk_boundaries: vec![0, 1],
            generated_ids: vec![10],
            next_token: 100,
            finish_reason: None,
            result_tx: tx,
        });

        // Modify next_token via mutable access
        if let Some(req) = scheduler.get_active_mut(1) {
            req.next_token = 999;
            req.generated_ids.push(42);
        }

        let req = scheduler.get_active_mut(1).unwrap();
        assert_eq!(req.next_token, 999);
        assert_eq!(req.generated_ids, vec![10, 42]);
    }
}
