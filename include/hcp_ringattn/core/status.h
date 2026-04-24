#pragma once

#include <sstream>
#include <string>
#include <utility>

namespace hcp_ringattn {

class Status {
 public:
  enum class Code {
    kOk = 0,
    kInvalidArgument = 1,
    kUnimplemented = 2,
    kInternal = 3,
  };

  Status() = default;

  static Status Ok() { return Status(); }

  template <typename... Args>
  static Status Error(Code code, Args&&... args) {
    return Status(code, BuildMessage(std::forward<Args>(args)...));
  }

  bool ok() const { return code_ == Code::kOk; }
  Code code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  template <typename... Args>
  static std::string BuildMessage(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
  }

  Status(Code code, std::string message)
      : code_(code), message_(std::move(message)) {}

  Code code_ = Code::kOk;
  std::string message_;
};

}  // namespace hcp_ringattn

