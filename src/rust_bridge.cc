#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <string>

#include "hcp_ringattn/core/ringattn_runtime.h"

#ifdef HCP_ENABLE_TORCH
#include <ATen/ATen.h>
#include <ATen/Context.h>
#endif

namespace {
std::string g_torch_smoke_message;
std::string g_torch_attention_smoke_message;
std::string g_torch_block_update_smoke_message;

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
