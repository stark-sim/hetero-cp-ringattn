//! Experimental single-layer self-driving decode ring.

use crate::model::attention::HcpRingAttentionBackend;
use crate::model::layers::DecoderLayer;
use crate::model::ModelError;
use tch::Tensor;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleLayerRingStats {
    pub domains: usize,
    pub hops: usize,
    pub visited_domains: Vec<usize>,
    pub local_partials: usize,
    pub q_projections: usize,
    pub q_projection_domain: usize,
    pub current_kv_projections: usize,
    pub current_kv_projection_domain: usize,
    pub current_kv_commits: usize,
    pub layer_finishes: usize,
    pub starter: usize,
    pub assignee: usize,
    pub finisher: usize,
}

#[derive(Debug)]
pub struct SingleLayerRingResult {
    pub attention_output: Tensor,
    pub hidden_states: Tensor,
    pub stats: SingleLayerRingStats,
}

/// Run one real decoder layer over already-sharded history KV.
///
/// This is deliberately an in-process experiment: it proves the tensor data
/// flow without introducing transport, runtime, admission, or retry behavior.
pub fn run_single_layer_ring(
    layer: &mut DecoderLayer,
    hidden_states: &Tensor,
    position_ids: &Tensor,
    history_shards: &mut [(Tensor, Tensor)],
    starter: usize,
    assignee: usize,
) -> Result<SingleLayerRingResult, ModelError> {
    let domains = history_shards.len();
    if domains == 0 {
        return Err(ModelError::Backend(
            "self-driving layer requires at least one domain".to_string(),
        ));
    }
    if starter >= domains || assignee >= domains {
        return Err(ModelError::Backend(format!(
            "self-driving route out of range: domains={domains}, starter={starter}, assignee={assignee}"
        )));
    }
    if hidden_states.size().get(1) != Some(&1) {
        return Err(ModelError::Backend(
            "self-driving layer requires one decode token".to_string(),
        ));
    }

    let residual = hidden_states.shallow_clone();
    let normalized = layer.input_layernorm.forward(hidden_states);
    let attention_output = {
        let ring = layer
            .attention
            .as_any_mut()
            .downcast_mut::<HcpRingAttentionBackend>()
            .ok_or_else(|| {
                ModelError::Backend(
                    "self-driving layer requires HcpRingAttentionBackend".to_string(),
                )
            })?;
        let q = ring.project_decode_q(&normalized, position_ids)?;
        let q_projections = 1;
        let mut current_kv_projections = 0;
        let mut current_kv_commits = 0;
        let mut accumulator: Option<(Tensor, Tensor)> = None;
        let mut visited_domains = Vec::with_capacity(domains);

        for step in 0..domains {
            let domain = (starter + step) % domains;
            visited_domains.push(domain);
            if domain == assignee {
                let (current_k, current_v) =
                    ring.project_decode_current_kv(&normalized, position_ids)?;
                current_kv_projections += 1;
                let committed_k = Tensor::cat(&[&history_shards[domain].0, &current_k], 2);
                let committed_v = Tensor::cat(&[&history_shards[domain].1, &current_v], 2);
                history_shards[domain] = (committed_k, committed_v);
                current_kv_commits += 1;
            }
            let (history_k, history_v) = &history_shards[domain];
            if history_k.size() != history_v.size() {
                return Err(ModelError::Backend(format!(
                    "domain {domain} has mismatched K/V shapes"
                )));
            }
            accumulator = Some(match accumulator {
                None => ring.decode_local_compact_partial(&q, history_k, history_v),
                Some((o, lse)) => {
                    ring.decode_merge_compact_partial(&q, &o, &lse, history_k, history_v)
                }
            });
        }

        let (o, _) = accumulator.expect("non-empty ring must produce an accumulator");
        (
            ring.project_decode_output(&o),
            visited_domains,
            q_projections,
            current_kv_projections,
            current_kv_commits,
        )
    };

    let (
        attention_output,
        visited_domains,
        q_projections,
        current_kv_projections,
        current_kv_commits,
    ) = attention_output;

    let post_attention = &attention_output + residual;
    let mlp_output = layer
        .mlp
        .forward(&layer.post_attention_layernorm.forward(&post_attention));
    let hidden_states = post_attention + mlp_output;
    let finisher = (starter + domains - 1) % domains;

    Ok(SingleLayerRingResult {
        attention_output,
        hidden_states,
        stats: SingleLayerRingStats {
            domains,
            hops: domains - 1,
            visited_domains,
            local_partials: domains,
            q_projections,
            q_projection_domain: starter,
            current_kv_projections,
            current_kv_projection_domain: assignee,
            current_kv_commits,
            layer_finishes: 1,
            starter,
            assignee,
            finisher,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cache::{ContiguousKvCache, KvCache};
    use crate::model::layers::{GqaAttention, Mlp, RmsNorm, RotaryEmbedding};
    use crate::model::model::LlamaModel;
    use crate::model::{ModelConfig, ModelWeights, WeightNames};
    use std::collections::HashMap;
    use tch::{Device, Kind, Tensor};

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: Some(1),
            intermediate_size: 64,
            vocab_size: 100,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            torch_dtype: Some("float32".to_string()),
            hidden_act: "silu".to_string(),
            max_position_embeddings: Some(128),
            attention_dropout: 0.0,
            bos_token_id: None,
            eos_token_id: None,
            use_cache: true,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: 1.0,
        }
    }

    fn deterministic_tensor(shape: &[i64], phase: f64, device: Device) -> Tensor {
        let numel = shape.iter().product::<i64>();
        ((Tensor::arange(numel, (Kind::Float, device)) + phase).sin() * 0.05).reshape(shape)
    }

    fn deterministic_weights(config: &ModelConfig, device: Device) -> ModelWeights {
        let hidden = config.hidden_size as i64;
        let intermediate = config.intermediate_size as i64;
        let q_width = (config.num_heads * config.head_dim()) as i64;
        let kv_width = (config.num_kv_heads() * config.head_dim()) as i64;
        let mut tensors = HashMap::new();
        tensors.insert(
            WeightNames::embedding().to_string(),
            deterministic_tensor(&[config.vocab_size as i64, hidden], 1.0, device),
        );
        tensors.insert(
            WeightNames::layer_norm().to_string(),
            Tensor::ones([hidden], (Kind::Float, device)),
        );
        tensors.insert(
            WeightNames::lm_head().to_string(),
            deterministic_tensor(&[config.vocab_size as i64, hidden], 2.0, device),
        );
        tensors.insert(
            WeightNames::rms_norm_weight(0),
            Tensor::ones([hidden], (Kind::Float, device)),
        );
        tensors.insert(
            WeightNames::post_attn_norm_weight(0),
            Tensor::ones([hidden], (Kind::Float, device)),
        );
        for (name, shape, phase) in [
            (WeightNames::q_proj_weight(0), vec![q_width, hidden], 3.0),
            (WeightNames::k_proj_weight(0), vec![kv_width, hidden], 4.0),
            (WeightNames::v_proj_weight(0), vec![kv_width, hidden], 5.0),
            (WeightNames::o_proj_weight(0), vec![hidden, q_width], 6.0),
            (
                WeightNames::gate_proj_weight(0),
                vec![intermediate, hidden],
                7.0,
            ),
            (
                WeightNames::up_proj_weight(0),
                vec![intermediate, hidden],
                8.0,
            ),
            (
                WeightNames::down_proj_weight(0),
                vec![hidden, intermediate],
                9.0,
            ),
        ] {
            tensors.insert(name, deterministic_tensor(&shape, phase, device));
        }
        ModelWeights { tensors }
    }

    fn reference_layer_output(
        config: &ModelConfig,
        weights: &crate::model::ModelWeights,
        hidden: &Tensor,
        position_ids: &Tensor,
        history_k: &Tensor,
        history_v: &Tensor,
    ) -> (Tensor, Tensor) {
        let device = hidden.device();
        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.unwrap(),
            config.rope_theta,
            device,
        );
        let attention = GqaAttention::from_weights(weights, 0, config, &rope).unwrap();
        let input_norm = RmsNorm::from_weights(
            weights,
            &WeightNames::rms_norm_weight(0),
            config.rms_norm_eps,
        )
        .unwrap();
        let post_norm = RmsNorm::from_weights(
            weights,
            &WeightNames::post_attn_norm_weight(0),
            config.rms_norm_eps,
        )
        .unwrap();
        let mlp = Mlp::from_weights(weights, 0).unwrap();
        let mut cache = ContiguousKvCache::new();
        let _ = cache.update(history_k, history_v).unwrap();

        let normed = input_norm.forward(hidden);
        let attention_output = attention
            .forward(&normed, position_ids, Some(&mut cache), None)
            .unwrap();
        let post_attention = &attention_output + hidden;
        let mlp_output = mlp.forward(&post_norm.forward(&post_attention));
        (attention_output, post_attention + mlp_output)
    }

    #[test]
    fn three_domain_uneven_ring_matches_reference_layer() {
        let device = Device::Cpu;
        let config = test_config();
        let weights = deterministic_weights(&config, device);
        let mut model = LlamaModel::from_weights(config.clone(), &weights, device, 3).unwrap();
        let mut layer = model.layers.remove(0);
        let hidden = deterministic_tensor(&[1, 1, config.hidden_size as i64], 10.0, device);
        let shard_lengths = [2_i64, 5, 3];
        let history_len = shard_lengths.iter().sum::<i64>();
        let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
        let mut shards = Vec::new();
        for &len in &shard_lengths {
            let shape = [
                1,
                config.num_kv_heads() as i64,
                len,
                config.head_dim() as i64,
            ];
            let domain = shards.len() as f64;
            shards.push((
                deterministic_tensor(&shape, 20.0 + domain, device),
                deterministic_tensor(&shape, 30.0 + domain, device),
            ));
        }

        let history_k = Tensor::cat(
            &shards
                .iter()
                .map(|(k, _)| k.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let history_v = Tensor::cat(
            &shards
                .iter()
                .map(|(_, v)| v.shallow_clone())
                .collect::<Vec<_>>(),
            2,
        );
        let result =
            run_single_layer_ring(&mut layer, &hidden, &position_ids, &mut shards, 1, 2).unwrap();
        assert_eq!(shards[0].0.size()[2], shard_lengths[0]);
        assert_eq!(shards[1].0.size()[2], shard_lengths[1]);
        assert_eq!(shards[2].0.size()[2], shard_lengths[2] + 1);
        let (reference_attention, reference) = reference_layer_output(
            &config,
            &weights,
            &hidden,
            &position_ids,
            &history_k,
            &history_v,
        );

        let mut single_model =
            LlamaModel::from_weights(config.clone(), &weights, device, 1).unwrap();
        let mut single_layer = single_model.layers.remove(0);
        let mut single_shard = vec![(history_k.shallow_clone(), history_v.shallow_clone())];
        let single = run_single_layer_ring(
            &mut single_layer,
            &hidden,
            &position_ids,
            &mut single_shard,
            0,
            0,
        )
        .unwrap();

        let max_diff = (&result.hidden_states - &reference)
            .abs()
            .max()
            .double_value(&[]);
        let single_reference_diff = (&single.hidden_states - &reference)
            .abs()
            .max()
            .double_value(&[]);
        let split_single_diff = (&result.hidden_states - &single.hidden_states)
            .abs()
            .max()
            .double_value(&[]);
        let attention_max_diff = (&result.attention_output - &reference_attention)
            .abs()
            .max()
            .double_value(&[]);
        let attention_mean_diff = (&result.attention_output - &reference_attention)
            .abs()
            .mean(Kind::Float)
            .double_value(&[]);
        eprintln!(
            "attention-max={attention_max_diff}, attention-mean={attention_mean_diff}, split-reference={max_diff}, single-reference={single_reference_diff}, split-single={split_single_diff}"
        );
        assert!(
            attention_max_diff < 1e-4,
            "max attention output diff: {attention_max_diff}"
        );
        assert!(
            split_single_diff < 1e-4,
            "split ring differs from single-block online softmax: {split_single_diff}"
        );
        // Synthetic weights are intentionally unscaled; O projection and MLP
        // amplify the much smaller attention difference measured above.
        assert!(max_diff < 2e-4, "max layer output diff: {max_diff}");
        assert_eq!(result.stats.domains, 3);
        assert_eq!(result.stats.hops, 2);
        assert_eq!(result.stats.local_partials, 3);
        assert_eq!(result.stats.q_projections, 1);
        assert_eq!(result.stats.q_projection_domain, 1);
        assert_eq!(result.stats.current_kv_projections, 1);
        assert_eq!(result.stats.current_kv_projection_domain, 2);
        assert_eq!(result.stats.current_kv_commits, 1);
        assert_eq!(result.stats.layer_finishes, 1);
        assert_eq!(result.stats.starter, 1);
        assert_eq!(result.stats.assignee, 2);
        assert_eq!(result.stats.finisher, 0);
    }

    #[test]
    fn arbitrary_domain_count_follows_successor_ring() {
        for domains in [1_usize, 2, 4] {
            let device = Device::Cpu;
            let config = test_config();
            let weights = deterministic_weights(&config, device);
            let mut model =
                LlamaModel::from_weights(config.clone(), &weights, device, domains).unwrap();
            let mut layer = model.layers.remove(0);
            let hidden = deterministic_tensor(
                &[1, 1, config.hidden_size as i64],
                40.0 + domains as f64,
                device,
            );
            let shard_lengths = (0..domains)
                .map(|domain| domain as i64 + 1)
                .collect::<Vec<_>>();
            let history_len = shard_lengths.iter().sum::<i64>();
            let position_ids = Tensor::from_slice(&[history_len]).unsqueeze(0);
            let mut shards = shard_lengths
                .iter()
                .enumerate()
                .map(|(domain, &len)| {
                    let shape = [
                        1,
                        config.num_kv_heads() as i64,
                        len,
                        config.head_dim() as i64,
                    ];
                    (
                        deterministic_tensor(&shape, 50.0 + domain as f64, device),
                        deterministic_tensor(&shape, 60.0 + domain as f64, device),
                    )
                })
                .collect::<Vec<_>>();
            let starter = domains - 1;
            let assignee = (starter + 1) % domains;
            let history_k = Tensor::cat(
                &shards
                    .iter()
                    .map(|(k, _)| k.shallow_clone())
                    .collect::<Vec<_>>(),
                2,
            );
            let history_v = Tensor::cat(
                &shards
                    .iter()
                    .map(|(_, v)| v.shallow_clone())
                    .collect::<Vec<_>>(),
                2,
            );

            let result = run_single_layer_ring(
                &mut layer,
                &hidden,
                &position_ids,
                &mut shards,
                starter,
                assignee,
            )
            .unwrap();
            let expected_route = (0..domains)
                .map(|step| (starter + step) % domains)
                .collect::<Vec<_>>();

            assert_eq!(result.stats.visited_domains, expected_route);
            assert_eq!(result.stats.domains, domains);
            assert_eq!(result.stats.hops, domains - 1);
            assert_eq!(result.stats.local_partials, domains);
            assert_eq!(result.stats.q_projections, 1);
            assert_eq!(result.stats.q_projection_domain, starter);
            assert_eq!(result.stats.current_kv_projections, 1);
            assert_eq!(result.stats.current_kv_projection_domain, assignee);
            assert_eq!(result.stats.current_kv_commits, 1);
            assert_eq!(result.stats.layer_finishes, 1);
            assert_eq!(result.stats.finisher, (starter + domains - 1) % domains);

            let (reference_attention, reference) = reference_layer_output(
                &config,
                &weights,
                &hidden,
                &position_ids,
                &history_k,
                &history_v,
            );
            let attention_diff = (&result.attention_output - reference_attention)
                .abs()
                .max()
                .double_value(&[]);
            let layer_diff = (&result.hidden_states - reference)
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                attention_diff < 1e-4,
                "N={domains} attention diff: {attention_diff}"
            );
            assert!(layer_diff < 2e-4, "N={domains} layer diff: {layer_diff}");
        }
    }
}
