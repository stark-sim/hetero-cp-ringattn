#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "hcp_ringattn/core/ringattn_runtime.h"

#ifdef HCP_ENABLE_TORCH
#include <ATen/ATen.h>
#include <ATen/Context.h>
#endif

namespace {
std::string g_torch_smoke_message;
std::string g_torch_attention_smoke_message;
std::string g_torch_block_update_smoke_message;
std::string g_torch_payload_block_smoke_message;
std::string g_torch_payload_online_smoke_message;
std::string g_torch_payload_chunk_smoke_message;
std::string g_torch_query_chunk_smoke_message;
std::string g_torch_query_chunk_output_smoke_message;

#ifdef HCP_ENABLE_TORCH
struct TorchDeviceSelection {
  at::Device device;
  int success_code;
};

bool ParseNonNegativeIndex(const std::string& value, int* index) {
  if (value.empty()) {
    return false;
  }
  int parsed = 0;
  for (char ch : value) {
    if (ch < '0' || ch > '9') {
      return false;
    }
    parsed = parsed * 10 + (ch - '0');
  }
  *index = parsed;
  return true;
}

bool SelectTorchDevice(const std::string& device_name, TorchDeviceSelection* selection) {
  if (device_name == "cpu") {
    *selection = {at::Device(at::kCPU), 1};
    return true;
  }
  if (device_name == "mps") {
    *selection = {at::Device(at::kMPS), 2};
    return true;
  }
  if (device_name == "cuda") {
    *selection = {at::Device(at::kCUDA), 3};
    return true;
  }
  const std::string cuda_prefix = "cuda:";
  if (device_name.rfind(cuda_prefix, 0) == 0) {
    int index = 0;
    if (!ParseNonNegativeIndex(device_name.substr(cuda_prefix.size()), &index)) {
      return false;
    }
    *selection = {at::Device(at::kCUDA, index), 3};
    return true;
  }
  return false;
}

bool DeviceMatches(const at::Tensor& tensor, const at::Device& device) {
  return device.is_cpu() ? tensor.is_cpu() : device.is_mps() ? tensor.is_mps() : tensor.is_cuda();
}

at::Tensor PayloadAttention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.sizes()[1]));
  auto k_by_head = k.permute({1, 0, 2});
  auto v_by_head = v.permute({1, 0, 2});
  auto scores = at::bmm(k_by_head, q.unsqueeze(2)).squeeze(2) * scale;
  auto weights = at::softmax(scores, -1);
  return at::bmm(weights.unsqueeze(1), v_by_head).squeeze(1);
}

at::Tensor PayloadChunkAttention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.sizes()[2]));
  auto q_by_head = q.permute({1, 0, 2});
  auto k_by_head = k.permute({1, 0, 2});
  auto v_by_head = v.permute({1, 0, 2});
  auto scores = at::bmm(q_by_head, k_by_head.transpose(1, 2)) * scale;
  auto weights = at::softmax(scores, -1);
  return at::bmm(weights, v_by_head).permute({1, 0, 2});
}

double TensorWeightedChecksum(const at::Tensor& tensor) {
  auto flat = tensor.contiguous().view({-1});
  const auto values = flat.data_ptr<float>();
  double checksum = 0.0;
  for (std::int64_t index = 0; index < flat.numel(); ++index) {
    checksum += static_cast<double>(values[index]) * static_cast<double>((index % 997) + 1);
  }
  return checksum;
}

int RunTorchAttentionBlockUpdates(int block_updates, std::string* message) {
  message->clear();
  if (block_updates <= 0) {
    *message = "block_updates must be positive";
    return -6;
  }
  const char* requested = std::getenv("HCP_TORCH_DEVICE");
  std::string device_name = requested == nullptr ? "cpu" : requested;
  TorchDeviceSelection selection{at::Device(at::kCPU), 1};
  if (!SelectTorchDevice(device_name, &selection)) {
    *message =
        "unsupported HCP_TORCH_DEVICE=" + device_name + "; expected cpu, mps, cuda, or cuda:N";
    return -4;
  }
  if (selection.device.is_cuda() && !at::hasCUDA()) {
    *message =
        "CUDA device name is valid, but CUDA backend is not available in the current libtorch "
        "process. Verify LIBTORCH/LIBTORCH_LIB point to a CUDA-enabled libtorch build and that "
        "libtorch_cuda and c10_cuda are linked/loaded.";
    return -5;
  }

  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto q_cpu = at::arange(32, cpu_options).reshape({4, 8}) / 17.0;
  auto k_cpu = at::arange(48, cpu_options).reshape({6, 8}) / 19.0;
  auto v_cpu = at::arange(48, cpu_options).reshape({6, 8}) / 23.0;

  const double scale = 1.0 / std::sqrt(8.0);
  float max_abs_err = 0.0F;
  float checksum = 0.0F;
  for (int update = 0; update < block_updates; ++update) {
    const double shift = static_cast<double>(update) / 101.0;
    auto q_update_cpu = q_cpu + shift;
    auto k_update_cpu = k_cpu + shift / 3.0;
    auto v_update_cpu = v_cpu + shift / 5.0;
    auto reference =
        at::matmul(at::softmax(at::matmul(q_update_cpu, k_update_cpu.transpose(0, 1)) * scale, -1),
                   v_update_cpu);

    auto q = q_update_cpu.to(selection.device);
    auto k = k_update_cpu.to(selection.device);
    auto v = v_update_cpu.to(selection.device);
    auto output = at::matmul(at::softmax(at::matmul(q, k.transpose(0, 1)) * scale, -1), v);
    if (!DeviceMatches(output, selection.device)) {
      *message = "attention output landed on unexpected device: " + output.device().str();
      return -1;
    }
    auto output_cpu = output.to(at::kCPU);
    max_abs_err = std::max(max_abs_err, at::abs(output_cpu - reference).max().item<float>());
    checksum += output_cpu.sum().item<float>();
    if (output.sizes()[0] != 4 || output.sizes()[1] != 8) {
      *message = "unexpected attention output shape";
      return -1;
    }
  }
  if (max_abs_err <= 1.0e-4F) {
    *message = "ok updates=" + std::to_string(block_updates) +
               " max_abs_err=" + std::to_string(max_abs_err) +
               " checksum=" + std::to_string(checksum);
    return selection.success_code;
  }
  *message = "attention mismatch updates=" + std::to_string(block_updates) +
             " max_abs_err=" + std::to_string(max_abs_err);
  return -1;
}

int RunTorchPayloadBlockSmoke(const std::uint8_t* payload,
                              std::size_t payload_len,
                              int block_len,
                              int num_heads,
                              int head_dim,
                              std::string* message) {
  message->clear();
  if (payload == nullptr) {
    *message = "payload pointer is null";
    return -6;
  }
  if (block_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    *message = "block_len, num_heads, and head_dim must be positive";
    return -6;
  }
  const auto block = static_cast<std::size_t>(block_len);
  const auto heads = static_cast<std::size_t>(num_heads);
  const auto dim = static_cast<std::size_t>(head_dim);
  const std::size_t values_per_tensor = block * heads * dim;
  const std::size_t expected_values = values_per_tensor * 2;
  const std::size_t expected_bytes = expected_values * sizeof(float);
  if (payload_len != expected_bytes) {
    *message = "payload byte size mismatch expected=" + std::to_string(expected_bytes) +
               " actual=" + std::to_string(payload_len);
    return -6;
  }

  const char* requested = std::getenv("HCP_TORCH_DEVICE");
  std::string device_name = requested == nullptr ? "cpu" : requested;
  TorchDeviceSelection selection{at::Device(at::kCPU), 1};
  if (!SelectTorchDevice(device_name, &selection)) {
    *message =
        "unsupported HCP_TORCH_DEVICE=" + device_name + "; expected cpu, mps, cuda, or cuda:N";
    return -4;
  }
  if (selection.device.is_cuda() && !at::hasCUDA()) {
    *message =
        "CUDA device name is valid, but CUDA backend is not available in the current libtorch "
        "process. Verify LIBTORCH/LIBTORCH_LIB point to a CUDA-enabled libtorch build and that "
        "libtorch_cuda and c10_cuda are linked/loaded.";
    return -5;
  }

  std::vector<float> values(expected_values);
  std::memcpy(values.data(), payload, expected_bytes);
  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto kv_cpu =
      at::from_blob(values.data(),
                    {2, static_cast<std::int64_t>(block_len), static_cast<std::int64_t>(num_heads),
                     static_cast<std::int64_t>(head_dim)},
                    cpu_options)
          .clone();
  auto q_cpu = (at::arange(num_heads * head_dim, cpu_options)
                    .reshape({static_cast<std::int64_t>(num_heads),
                              static_cast<std::int64_t>(head_dim)}) /
                31.0) +
               0.125;
  auto k_cpu = kv_cpu[0];
  auto v_cpu = kv_cpu[1];
  auto reference = PayloadAttention(q_cpu, k_cpu, v_cpu);

  auto q = q_cpu.to(selection.device);
  auto k = k_cpu.to(selection.device);
  auto v = v_cpu.to(selection.device);
  auto output = PayloadAttention(q, k, v);
  if (!DeviceMatches(output, selection.device)) {
    *message = "payload attention output landed on unexpected device: " + output.device().str();
    return -1;
  }
  if (output.sizes()[0] != num_heads || output.sizes()[1] != head_dim) {
    *message = "unexpected payload attention output shape";
    return -1;
  }
  auto output_cpu = output.to(at::kCPU);
  const float max_abs_err = at::abs(output_cpu - reference).max().item<float>();
  const float checksum = output_cpu.sum().item<float>();
  if (max_abs_err <= 1.0e-4F) {
    *message = "ok block_len=" + std::to_string(block_len) +
               " num_heads=" + std::to_string(num_heads) +
               " head_dim=" + std::to_string(head_dim) +
               " max_abs_err=" + std::to_string(max_abs_err) +
               " checksum=" + std::to_string(checksum);
    return selection.success_code;
  }
  *message = "payload attention mismatch block_len=" + std::to_string(block_len) +
             " max_abs_err=" + std::to_string(max_abs_err);
  return -1;
}

int RunTorchPayloadOnlineSmoke(const std::uint8_t* payload,
                               std::size_t payload_len,
                               const int* block_lens,
                               std::size_t block_count,
                               int num_heads,
                               int head_dim,
                               std::string* message) {
  message->clear();
  if (payload == nullptr || block_lens == nullptr) {
    *message = "payload or block_lens pointer is null";
    return -6;
  }
  if (block_count == 0 || num_heads <= 0 || head_dim <= 0) {
    *message = "block_count, num_heads, and head_dim must be positive";
    return -6;
  }
  std::size_t token_count = 0;
  for (std::size_t index = 0; index < block_count; ++index) {
    if (block_lens[index] <= 0) {
      *message = "all block_lens entries must be positive";
      return -6;
    }
    token_count += static_cast<std::size_t>(block_lens[index]);
  }
  const std::size_t expected_values =
      token_count * static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(head_dim) * 2;
  const std::size_t expected_bytes = expected_values * sizeof(float);
  if (payload_len != expected_bytes) {
    *message = "online payload byte size mismatch expected=" + std::to_string(expected_bytes) +
               " actual=" + std::to_string(payload_len);
    return -6;
  }

  const char* requested = std::getenv("HCP_TORCH_DEVICE");
  std::string device_name = requested == nullptr ? "cpu" : requested;
  TorchDeviceSelection selection{at::Device(at::kCPU), 1};
  if (!SelectTorchDevice(device_name, &selection)) {
    *message =
        "unsupported HCP_TORCH_DEVICE=" + device_name + "; expected cpu, mps, cuda, or cuda:N";
    return -4;
  }
  if (selection.device.is_cuda() && !at::hasCUDA()) {
    *message =
        "CUDA device name is valid, but CUDA backend is not available in the current libtorch "
        "process. Verify LIBTORCH/LIBTORCH_LIB point to a CUDA-enabled libtorch build and that "
        "libtorch_cuda and c10_cuda are linked/loaded.";
    return -5;
  }

  std::vector<float> values(expected_values);
  std::memcpy(values.data(), payload, expected_bytes);
  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto device_options = at::TensorOptions().dtype(at::kFloat).device(selection.device);
  auto q_cpu = (at::arange(num_heads * head_dim, cpu_options)
                    .reshape({static_cast<std::int64_t>(num_heads),
                              static_cast<std::int64_t>(head_dim)}) /
                31.0) +
               0.125;
  auto q = q_cpu.to(selection.device);
  auto running_max =
      at::full({num_heads}, -std::numeric_limits<float>::infinity(), device_options);
  auto running_sum = at::zeros({num_heads}, device_options);
  auto output = at::zeros({num_heads, head_dim}, device_options);
  std::vector<at::Tensor> k_refs;
  std::vector<at::Tensor> v_refs;
  k_refs.reserve(block_count);
  v_refs.reserve(block_count);

  std::size_t value_offset = 0;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
    const int block_len = block_lens[block_index];
    const std::size_t block_values = static_cast<std::size_t>(block_len) *
                                     static_cast<std::size_t>(num_heads) *
                                     static_cast<std::size_t>(head_dim) * 2;
    auto kv_cpu =
        at::from_blob(values.data() + value_offset,
                      {2, static_cast<std::int64_t>(block_len),
                       static_cast<std::int64_t>(num_heads),
                       static_cast<std::int64_t>(head_dim)},
                      cpu_options)
            .clone();
    value_offset += block_values;
    auto k_cpu = kv_cpu[0];
    auto v_cpu = kv_cpu[1];
    k_refs.push_back(k_cpu);
    v_refs.push_back(v_cpu);

    auto k = k_cpu.to(selection.device);
    auto v = v_cpu.to(selection.device);
    auto k_by_head = k.permute({1, 0, 2});
    auto v_by_head = v.permute({1, 0, 2});
    auto scores = at::bmm(k_by_head, q.unsqueeze(2)).squeeze(2) * scale;
    auto local_max = std::get<0>(scores.max(1));
    auto weights = at::exp(scores - local_max.unsqueeze(1));
    auto local_sum = weights.sum(1);
    auto local_pv = at::bmm(weights.unsqueeze(1), v_by_head).squeeze(1);

    auto new_max = at::maximum(running_max, local_max);
    auto exp_prev = at::exp(running_max - new_max);
    auto exp_local = at::exp(local_max - new_max);
    auto new_sum = exp_prev * running_sum + exp_local * local_sum;
    output = (exp_prev.unsqueeze(1) * running_sum.unsqueeze(1) * output +
              exp_local.unsqueeze(1) * local_pv) /
             new_sum.unsqueeze(1);
    running_max = new_max;
    running_sum = new_sum;
  }

  if (!DeviceMatches(output, selection.device)) {
    *message = "online output landed on unexpected device: " + output.device().str();
    return -1;
  }
  auto reference = PayloadAttention(q_cpu, at::cat(k_refs, 0), at::cat(v_refs, 0));
  auto output_cpu = output.to(at::kCPU);
  const float max_abs_err = at::abs(output_cpu - reference).max().item<float>();
  const float checksum = output_cpu.sum().item<float>();
  if (max_abs_err <= 1.0e-4F) {
    *message = "ok blocks=" + std::to_string(block_count) +
               " tokens=" + std::to_string(token_count) +
               " max_abs_err=" + std::to_string(max_abs_err) +
               " checksum=" + std::to_string(checksum);
    return selection.success_code;
  }
  *message = "online payload mismatch blocks=" + std::to_string(block_count) +
             " tokens=" + std::to_string(token_count) +
             " max_abs_err=" + std::to_string(max_abs_err);
  return -1;
}

int RunTorchPayloadChunkSmokeWithQCpu(const at::Tensor& q_cpu,
                                      const std::uint8_t* payload,
                                      std::size_t payload_len,
                                      const int* block_lens,
                                      std::size_t block_count,
                                      int query_len,
                                      int num_heads,
                                      int head_dim,
                                      double* output_checksum,
                                      double* max_abs_err_out,
                                      std::size_t* output_values,
                                      std::string* message) {
  message->clear();
  if (payload == nullptr || block_lens == nullptr) {
    *message = "payload or block_lens pointer is null";
    return -6;
  }
  if (block_count == 0 || query_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    *message = "block_count, query_len, num_heads, and head_dim must be positive";
    return -6;
  }
  if (q_cpu.sizes().size() != 3 || q_cpu.sizes()[0] != query_len ||
      q_cpu.sizes()[1] != num_heads || q_cpu.sizes()[2] != head_dim || !q_cpu.is_cpu()) {
    *message = "q chunk tensor shape/device mismatch";
    return -6;
  }
  std::size_t token_count = 0;
  for (std::size_t index = 0; index < block_count; ++index) {
    if (block_lens[index] <= 0) {
      *message = "all block_lens entries must be positive";
      return -6;
    }
    token_count += static_cast<std::size_t>(block_lens[index]);
  }
  const std::size_t expected_values =
      token_count * static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(head_dim) * 2;
  const std::size_t expected_bytes = expected_values * sizeof(float);
  if (payload_len != expected_bytes) {
    *message = "chunk payload byte size mismatch expected=" + std::to_string(expected_bytes) +
               " actual=" + std::to_string(payload_len);
    return -6;
  }

  const char* requested = std::getenv("HCP_TORCH_DEVICE");
  std::string device_name = requested == nullptr ? "cpu" : requested;
  TorchDeviceSelection selection{at::Device(at::kCPU), 1};
  if (!SelectTorchDevice(device_name, &selection)) {
    *message =
        "unsupported HCP_TORCH_DEVICE=" + device_name + "; expected cpu, mps, cuda, or cuda:N";
    return -4;
  }
  if (selection.device.is_cuda() && !at::hasCUDA()) {
    *message =
        "CUDA device name is valid, but CUDA backend is not available in the current libtorch "
        "process. Verify LIBTORCH/LIBTORCH_LIB point to a CUDA-enabled libtorch build and that "
        "libtorch_cuda and c10_cuda are linked/loaded.";
    return -5;
  }

  std::vector<float> values(expected_values);
  std::memcpy(values.data(), payload, expected_bytes);
  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto device_options = at::TensorOptions().dtype(at::kFloat).device(selection.device);
  auto q = q_cpu.to(selection.device);
  auto q_by_head = q.permute({1, 0, 2});
  auto running_max =
      at::full({num_heads, query_len}, -std::numeric_limits<float>::infinity(), device_options);
  auto running_sum = at::zeros({num_heads, query_len}, device_options);
  auto output_by_head = at::zeros({num_heads, query_len, head_dim}, device_options);
  std::vector<at::Tensor> k_refs;
  std::vector<at::Tensor> v_refs;
  k_refs.reserve(block_count);
  v_refs.reserve(block_count);

  std::size_t value_offset = 0;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
    const int block_len = block_lens[block_index];
    const std::size_t block_values = static_cast<std::size_t>(block_len) *
                                     static_cast<std::size_t>(num_heads) *
                                     static_cast<std::size_t>(head_dim) * 2;
    auto kv_cpu =
        at::from_blob(values.data() + value_offset,
                      {2, static_cast<std::int64_t>(block_len),
                       static_cast<std::int64_t>(num_heads),
                       static_cast<std::int64_t>(head_dim)},
                      cpu_options)
            .clone();
    value_offset += block_values;
    auto k_cpu = kv_cpu[0];
    auto v_cpu = kv_cpu[1];
    k_refs.push_back(k_cpu);
    v_refs.push_back(v_cpu);

    auto k = k_cpu.to(selection.device);
    auto v = v_cpu.to(selection.device);
    auto k_by_head = k.permute({1, 0, 2});
    auto v_by_head = v.permute({1, 0, 2});
    auto scores = at::bmm(q_by_head, k_by_head.transpose(1, 2)) * scale;
    auto local_max = std::get<0>(scores.max(2));
    auto weights = at::exp(scores - local_max.unsqueeze(2));
    auto local_sum = weights.sum(2);
    auto local_pv = at::bmm(weights, v_by_head);

    auto new_max = at::maximum(running_max, local_max);
    auto exp_prev = at::exp(running_max - new_max);
    auto exp_local = at::exp(local_max - new_max);
    auto new_sum = exp_prev * running_sum + exp_local * local_sum;
    output_by_head = (exp_prev.unsqueeze(2) * running_sum.unsqueeze(2) * output_by_head +
                      exp_local.unsqueeze(2) * local_pv) /
                     new_sum.unsqueeze(2);
    running_max = new_max;
    running_sum = new_sum;
  }

  auto output = output_by_head.permute({1, 0, 2});
  if (!DeviceMatches(output, selection.device)) {
    *message = "chunk output landed on unexpected device: " + output.device().str();
    return -1;
  }
  auto reference = PayloadChunkAttention(q_cpu, at::cat(k_refs, 0), at::cat(v_refs, 0));
  auto output_cpu = output.to(at::kCPU);
  const float max_abs_err = at::abs(output_cpu - reference).max().item<float>();
  const double checksum = TensorWeightedChecksum(output_cpu);
  if (output_checksum != nullptr) {
    *output_checksum = checksum;
  }
  if (max_abs_err_out != nullptr) {
    *max_abs_err_out = static_cast<double>(max_abs_err);
  }
  if (output_values != nullptr) {
    *output_values = static_cast<std::size_t>(output_cpu.numel());
  }
  if (max_abs_err <= 1.0e-4F) {
    *message = "ok blocks=" + std::to_string(block_count) +
               " query_len=" + std::to_string(query_len) +
               " tokens=" + std::to_string(token_count) +
               " max_abs_err=" + std::to_string(max_abs_err) +
               " checksum=" + std::to_string(checksum);
    return selection.success_code;
  }
  *message = "chunk payload mismatch blocks=" + std::to_string(block_count) +
             " query_len=" + std::to_string(query_len) +
             " tokens=" + std::to_string(token_count) +
             " max_abs_err=" + std::to_string(max_abs_err);
  return -1;
}

int RunTorchPayloadChunkSmoke(const std::uint8_t* payload,
                              std::size_t payload_len,
                              const int* block_lens,
                              std::size_t block_count,
                              int query_len,
                              int num_heads,
                              int head_dim,
                              std::string* message) {
  message->clear();
  if (query_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    *message = "query_len, num_heads, and head_dim must be positive";
    return -6;
  }
  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto q_cpu = (at::arange(query_len * num_heads * head_dim, cpu_options)
                    .reshape({static_cast<std::int64_t>(query_len),
                              static_cast<std::int64_t>(num_heads),
                              static_cast<std::int64_t>(head_dim)}) /
                37.0) +
               0.0625;
  return RunTorchPayloadChunkSmokeWithQCpu(q_cpu, payload, payload_len, block_lens, block_count,
                                           query_len, num_heads, head_dim, nullptr, nullptr,
                                           nullptr, message);
}

int RunTorchQueryChunkSmoke(const std::uint8_t* q_payload,
                            std::size_t q_payload_len,
                            const std::uint8_t* kv_payload,
                            std::size_t kv_payload_len,
                            const int* block_lens,
                            std::size_t block_count,
                            int query_len,
                            int num_heads,
                            int head_dim,
                            double* output_checksum,
                            double* max_abs_err,
                            std::size_t* output_values,
                            std::string* message) {
  message->clear();
  if (q_payload == nullptr) {
    *message = "q payload pointer is null";
    return -6;
  }
  if (query_len <= 0 || num_heads <= 0 || head_dim <= 0) {
    *message = "query_len, num_heads, and head_dim must be positive";
    return -6;
  }
  const std::size_t expected_q_values = static_cast<std::size_t>(query_len) *
                                        static_cast<std::size_t>(num_heads) *
                                        static_cast<std::size_t>(head_dim);
  const std::size_t expected_q_bytes = expected_q_values * sizeof(float);
  if (q_payload_len != expected_q_bytes) {
    *message = "q payload byte size mismatch expected=" + std::to_string(expected_q_bytes) +
               " actual=" + std::to_string(q_payload_len);
    return -6;
  }

  std::vector<float> q_values(expected_q_values);
  std::memcpy(q_values.data(), q_payload, expected_q_bytes);
  auto cpu_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto q_cpu = at::from_blob(q_values.data(),
                             {static_cast<std::int64_t>(query_len),
                              static_cast<std::int64_t>(num_heads),
                              static_cast<std::int64_t>(head_dim)},
                             cpu_options)
                   .clone();
  return RunTorchPayloadChunkSmokeWithQCpu(q_cpu, kv_payload, kv_payload_len, block_lens,
                                           block_count, query_len, num_heads, head_dim,
                                           output_checksum, max_abs_err, output_values, message);
}
#endif
}

extern "C" const char* hcp_ringattn_torch_smoke_message() {
  return g_torch_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_attention_smoke_message() {
  return g_torch_attention_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_block_update_smoke_message() {
  return g_torch_block_update_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_payload_block_smoke_message() {
  return g_torch_payload_block_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_payload_online_smoke_message() {
  return g_torch_payload_online_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_payload_chunk_smoke_message() {
  return g_torch_payload_chunk_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_query_chunk_smoke_message() {
  return g_torch_query_chunk_smoke_message.c_str();
}

extern "C" const char* hcp_ringattn_torch_query_chunk_output_smoke_message() {
  return g_torch_query_chunk_output_smoke_message.c_str();
}

extern "C" int hcp_ringattn_cxx_smoke_domain_count() {
  try {
    hcp_ringattn::RingAttnConfig global_config;
    global_config.global_seq_len = 1024;
    global_config.num_heads = 8;
    global_config.head_dim = 64;
    global_config.domains = {
        {"domain-0", "127.0.0.1", 26001, 512, 128, "cpu"},
        {"domain-1", "127.0.0.1", 26002, 256, 64, "cpu"},
        {"domain-2", "127.0.0.1", 26003, 256, 64, "cpu"},
    };

    int ok = 0;
    for (const auto& domain : global_config.domains) {
      auto runtime = hcp_ringattn::CreateRingAttnRuntime(domain.device);
      hcp_ringattn::RingAttnSoftmaxState initial_state;
      hcp_ringattn::RingAttnSoftmaxState out_state;
      auto status = runtime->Init(domain, global_config);
      if (!status.ok()) {
        return -1;
      }
      status = runtime->RunRingAttentionLayer(0, initial_state, &out_state);
      if (!status.ok()) {
        return -2;
      }
      status = runtime->Shutdown();
      if (!status.ok()) {
        return -3;
      }
      ++ok;
    }
    return ok;
  } catch (const std::exception&) {
    return -10;
  } catch (...) {
    return -11;
  }
}

extern "C" int hcp_ringattn_torch_smoke() {
#ifdef HCP_ENABLE_TORCH
  try {
    g_torch_smoke_message.clear();
    const char* requested = std::getenv("HCP_TORCH_DEVICE");
    std::string device_name = requested == nullptr ? "cpu" : requested;
    TorchDeviceSelection selection{at::Device(at::kCPU), 1};
    if (!SelectTorchDevice(device_name, &selection)) {
      g_torch_smoke_message =
          "unsupported HCP_TORCH_DEVICE=" + device_name + "; expected cpu, mps, cuda, or cuda:N";
      return -4;
    }
    if (selection.device.is_cuda() && !at::hasCUDA()) {
      g_torch_smoke_message =
          "CUDA device name is valid, but CUDA backend is not available in the current libtorch "
          "process. Verify LIBTORCH/LIBTORCH_LIB point to a CUDA-enabled libtorch build and that "
          "libtorch_cuda and c10_cuda are linked/loaded.";
      return -5;
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(selection.device);
    auto a = at::ones({2, 2}, options);
    auto b = at::eye(2, options);
    auto c = at::matmul(a, b);
    if (c.sizes()[0] == 2 && c.sizes()[1] == 2 && DeviceMatches(c, selection.device)) {
      g_torch_smoke_message = "ok";
      return selection.success_code;
    }
    g_torch_smoke_message = "unexpected tensor shape or device: " + c.device().str();
    return -1;
  } catch (const std::exception& exc) {
    g_torch_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_smoke_message = "unknown exception";
    return -3;
  }
#else
  g_torch_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_attention_smoke() {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchAttentionBlockUpdates(1, &g_torch_attention_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_attention_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_attention_smoke_message = "unknown exception";
    return -3;
  }
#else
  g_torch_attention_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_block_update_smoke(int block_updates) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchAttentionBlockUpdates(block_updates, &g_torch_block_update_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_block_update_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_block_update_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)block_updates;
  g_torch_block_update_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_payload_block_smoke(const std::uint8_t* payload,
                                                       std::size_t payload_len,
                                                       int block_len,
                                                       int num_heads,
                                                       int head_dim) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchPayloadBlockSmoke(payload, payload_len, block_len, num_heads, head_dim,
                                     &g_torch_payload_block_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_payload_block_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_payload_block_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)payload;
  (void)payload_len;
  (void)block_len;
  (void)num_heads;
  (void)head_dim;
  g_torch_payload_block_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_payload_online_smoke(const std::uint8_t* payload,
                                                        std::size_t payload_len,
                                                        const int* block_lens,
                                                        std::size_t block_count,
                                                        int num_heads,
                                                        int head_dim) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchPayloadOnlineSmoke(payload, payload_len, block_lens, block_count, num_heads,
                                      head_dim, &g_torch_payload_online_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_payload_online_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_payload_online_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)payload;
  (void)payload_len;
  (void)block_lens;
  (void)block_count;
  (void)num_heads;
  (void)head_dim;
  g_torch_payload_online_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_payload_chunk_smoke(const std::uint8_t* payload,
                                                       std::size_t payload_len,
                                                       const int* block_lens,
                                                       std::size_t block_count,
                                                       int query_len,
                                                       int num_heads,
                                                       int head_dim) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchPayloadChunkSmoke(payload, payload_len, block_lens, block_count, query_len,
                                     num_heads, head_dim, &g_torch_payload_chunk_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_payload_chunk_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_payload_chunk_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)payload;
  (void)payload_len;
  (void)block_lens;
  (void)block_count;
  (void)query_len;
  (void)num_heads;
  (void)head_dim;
  g_torch_payload_chunk_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_query_chunk_smoke(const std::uint8_t* q_payload,
                                                     std::size_t q_payload_len,
                                                     const std::uint8_t* kv_payload,
                                                     std::size_t kv_payload_len,
                                                     const int* block_lens,
                                                     std::size_t block_count,
                                                     int query_len,
                                                     int num_heads,
                                                     int head_dim) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchQueryChunkSmoke(q_payload, q_payload_len, kv_payload, kv_payload_len,
                                   block_lens, block_count, query_len, num_heads, head_dim, nullptr,
                                   nullptr, nullptr,
                                   &g_torch_query_chunk_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_query_chunk_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_query_chunk_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)q_payload;
  (void)q_payload_len;
  (void)kv_payload;
  (void)kv_payload_len;
  (void)block_lens;
  (void)block_count;
  (void)query_len;
  (void)num_heads;
  (void)head_dim;
  g_torch_query_chunk_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}

extern "C" int hcp_ringattn_torch_query_chunk_output_smoke(const std::uint8_t* q_payload,
                                                            std::size_t q_payload_len,
                                                            const std::uint8_t* kv_payload,
                                                            std::size_t kv_payload_len,
                                                            const int* block_lens,
                                                            std::size_t block_count,
                                                            int query_len,
                                                            int num_heads,
                                                            int head_dim,
                                                            double* output_checksum,
                                                            double* max_abs_err,
                                                            std::size_t* output_values) {
#ifdef HCP_ENABLE_TORCH
  try {
    return RunTorchQueryChunkSmoke(q_payload, q_payload_len, kv_payload, kv_payload_len,
                                   block_lens, block_count, query_len, num_heads, head_dim,
                                   output_checksum, max_abs_err, output_values,
                                   &g_torch_query_chunk_output_smoke_message);
  } catch (const std::exception& exc) {
    g_torch_query_chunk_output_smoke_message = exc.what();
    return -2;
  } catch (...) {
    g_torch_query_chunk_output_smoke_message = "unknown exception";
    return -3;
  }
#else
  (void)q_payload;
  (void)q_payload_len;
  (void)kv_payload;
  (void)kv_payload_len;
  (void)block_lens;
  (void)block_count;
  (void)query_len;
  (void)num_heads;
  (void)head_dim;
  (void)output_checksum;
  (void)max_abs_err;
  (void)output_values;
  g_torch_query_chunk_output_smoke_message = "HCP_ENABLE_TORCH is not enabled";
  return 0;
#endif
}
