#include <cstdlib>
#include <exception>
#include <string>

#include "hcp_ringattn/core/ringattn_runtime.h"

#ifdef HCP_ENABLE_TORCH
#include <ATen/ATen.h>
#endif

namespace {
std::string g_torch_smoke_message;

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
#endif
}

extern "C" const char* hcp_ringattn_torch_smoke_message() {
  return g_torch_smoke_message.c_str();
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

    auto options = at::TensorOptions().dtype(at::kFloat).device(selection.device);
    auto a = at::ones({2, 2}, options);
    auto b = at::eye(2, options);
    auto c = at::matmul(a, b);
    const bool expected_device =
        selection.device.is_cpu() ? c.is_cpu()
        : selection.device.is_mps() ? c.is_mps()
                                    : c.is_cuda();
    if (c.sizes()[0] == 2 && c.sizes()[1] == 2 && expected_device) {
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
