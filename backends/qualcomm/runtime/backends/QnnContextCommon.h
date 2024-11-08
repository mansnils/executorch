/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>

#include <memory>
namespace executorch {
namespace backends {
namespace qnn {
class QnnContext {
 public:
  explicit QnnContext(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnDevice* device,
      QnnBackendCache* cache)
      : handle_(nullptr),
        implementation_(implementation),
        backend_(backend),
        device_(device),
        cache_(cache) {}

  virtual ~QnnContext();
  executorch::runtime::Error Configure();

  Qnn_ContextHandle_t GetHandle() const {
    return handle_;
  }

  std::string GetGraphName() {
    return cache_->GetGraphName();
  }

  std::vector<Qnn_Tensor_t> GetGraphInputs() {
    return cache_->GetGraphInputs();
  }
  std::vector<Qnn_Tensor_t> GetGraphOutputs() {
    return cache_->GetGraphOutputs();
  }
  QnnBackendCache::CacheState GetCacheState() const {
    return cache_->GetCacheState();
  };

  executorch::runtime::Error GetContextBinary(
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

 protected:
  virtual executorch::runtime::Error MakeConfig(
      std::vector<const QnnContext_Config_t*>& config) {
    return executorch::runtime::Error::Ok;
  };
  virtual executorch::runtime::Error AfterConfigure() {
    return executorch::runtime::Error::Ok;
  };

 private:
  Qnn_ContextHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnBackend* backend_;
  QnnDevice* device_;
  QnnBackendCache* cache_;
  std::vector<char> binary_buffer_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
