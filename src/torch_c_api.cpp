/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#if defined(__LIBTORCH)

#include <ATen/Parallel.h>
#if defined(__LIBTORCH_CUDA)
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#endif
#include <c10/core/DeviceGuard.h>
#include <torch/csrc/api/include/torch/cuda.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

#include "offload/offload_library.h"

#if defined(__OPENBLAS)
#include <cblas.h>
#endif

#include <cassert>

#include <cfenv>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__OPENBLAS)
// PyTorch's oneMKL batch ABI is not compatible with OpenBLAS's same-named
// entry points. Expand grouped GEMMs into the portable CBLAS interface.
extern "C" void
cblas_sgemm_batch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE *trans_a,
                  const CBLAS_TRANSPOSE *trans_b, const int *m, const int *n,
                  const int *k, const float *alpha, const float **a,
                  const int *lda, const float **b, const int *ldb,
                  const float *beta, float **c, const int *ldc,
                  const int group_count, const int *group_size) {
  int offset = 0;
  for (int group = 0; group < group_count; ++group) {
    for (int operation = 0; operation < group_size[group]; ++operation) {
      const int index = offset + operation;
      cblas_sgemm(order, trans_a[group], trans_b[group], m[group], n[group],
                  k[group], alpha[group], a[index], lda[group], b[index],
                  ldb[group], beta[group], c[index], ldc[group]);
    }
    offset += group_size[group];
  }
}

extern "C" void
cblas_dgemm_batch(const CBLAS_ORDER order, const CBLAS_TRANSPOSE *trans_a,
                  const CBLAS_TRANSPOSE *trans_b, const int *m, const int *n,
                  const int *k, const double *alpha, const double **a,
                  const int *lda, const double **b, const int *ldb,
                  const double *beta, double **c, const int *ldc,
                  const int group_count, const int *group_size) {
  int offset = 0;
  for (int group = 0; group < group_count; ++group) {
    for (int operation = 0; operation < group_size[group]; ++operation) {
      const int index = offset + operation;
      cblas_dgemm(order, trans_a[group], trans_b[group], m[group], n[group],
                  k[group], alpha[group], a[index], lda[group], b[index],
                  ldb[group], beta[group], c[index], ldc[group]);
    }
    offset += group_size[group];
  }
}
#endif

typedef torch::Tensor torch_c_tensor_t;
typedef c10::Dict<std::string, torch::Tensor> torch_c_dict_t;
typedef torch::jit::Module torch_c_model_t;

class TorchFloatingPointMaskGuard {
public:
  TorchFloatingPointMaskGuard() : active_(std::feholdexcept(&env_) == 0) {}
  ~TorchFloatingPointMaskGuard() {
    if (active_) {
      std::feclearexcept(FE_ALL_EXCEPT);
      std::fesetenv(&env_);
    }
  }

private:
  std::fenv_t env_;
  bool active_;
};

/*******************************************************************************
 * \brief Aborts the run with a message instead of letting a C++ exception
 *        unwind across the BIND(C) boundary.
 ******************************************************************************/
[[noreturn]] static void torch_c_api_abort(const std::string &message) {
  std::fprintf(stderr, "TORCH| %s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
}

static void torch_c_check(const bool condition, const char *message) {
  if (!condition) {
    torch_c_api_abort(message);
  }
}

// Wrap the body of every BIND(C) entry point so Torch exceptions become a
// controlled CP2K abort instead of undefined behaviour.
#define TORCH_C_TRY try {
#define TORCH_C_CATCH(NAME)                                                    \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    torch_c_api_abort(std::string(NAME) + ": " + e.what());                    \
  }                                                                            \
  catch (...) {                                                                \
    torch_c_api_abort(std::string(NAME) + ": unknown exception");              \
  }

/*******************************************************************************
 * \brief Internal helper for selecting the CUDA device when available.
 * \author Ole Schuett
 ******************************************************************************/
static bool use_cuda_if_available = true;

static void enable_expandable_cuda_segments_if_unconfigured() {
#if defined(__LIBTORCH_CUDA)
  static const bool initialized = []() {
    const char *legacy_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
    const char *config = std::getenv("PYTORCH_ALLOC_CONF");
    if ((legacy_config == nullptr || legacy_config[0] == '\0') &&
        (config == nullptr || config[0] == '\0')) {
      c10::cuda::CUDACachingAllocator::setAllocatorSettings(
          "expandable_segments:True");
    }
    return true;
  }();
  (void)initialized;
#endif
}

static bool get_positive_int_env(const char *name, int &value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  char *end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

static void initialize_torch_threads_from_env() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;

  int num_threads = 0;
  if (get_positive_int_env("CP2K_TORCH_NUM_THREADS", num_threads)) {
    at::set_num_threads(num_threads);
  }
  if (get_positive_int_env("CP2K_TORCH_NUM_INTEROP_THREADS", num_threads)) {
    at::set_num_interop_threads(num_threads);
  }
}

static torch::Device get_device() {
  initialize_torch_threads_from_env();
  if (!use_cuda_if_available || !torch::cuda::is_available()) {
    return torch::kCPU;
  }
  enable_expandable_cuda_segments_if_unconfigured();
  const auto device_count = torch::cuda::device_count();
  if (device_count <= 0) {
    return torch::kCPU;
  }
  const int chosen_device = offload_get_chosen_device();
  const int device = (chosen_device >= 0) ? chosen_device : 0;
  torch_c_check(device < device_count,
                "Torch CUDA device index is out of range");
  return torch::Device(torch::kCUDA, device);
}

static torch::Device get_device_with_guard(c10::OptionalDeviceGuard &guard) {
  const auto device = get_device();
  if (device.is_cuda()) {
    guard.reset_device(device);
  }
  return device;
}

static bool use_batched_gradient_readback(const torch::Tensor &tensor) {
#if defined(__LIBTORCH_CUDA)
  return tensor.is_cuda() &&
         at::cuda::getDeviceProperties(tensor.device().index())->integrated;
#else
  (void)tensor;
  return false;
#endif
}

static void set_jit_fusion_strategy() {
  // JIT Fusion strategy optimization, hardcode dynamic 10, see also
  // https://github.com/mir-group/pair_nequip_allegro.git
  torch::jit::FusionStrategy strategy = {
      {torch::jit::FusionBehavior::DYNAMIC, 10}};
  torch::jit::setFusionStrategy(strategy);
}

static void copy_string_to_c_buffer(const std::string &source, char **content,
                                    int *length) {
  *length = source.length();
  *content = (char *)malloc(source.length() + 1); // +1 for null terminator
  strcpy(*content, source.c_str());
}

static bool can_load_directly_to_device(const torch::Device &device) {
  return !device.is_cuda() || device.index() == 0 ||
         torch::cuda::device_count() == 1;
}

static torch::jit::Module load_module_for_device(
    const char *filename, const torch::Device &device,
    std::unordered_map<std::string, std::string> *extra_files = nullptr,
    const bool load_on_cpu = false) {
  if (!load_on_cpu && can_load_directly_to_device(device)) {
    if (extra_files != nullptr) {
      return torch::jit::load(filename, device, *extra_files);
    }
    return torch::jit::load(filename, device);
  }
  auto model = (extra_files != nullptr)
                   ? torch::jit::load(filename, torch::kCPU, *extra_files)
                   : torch::jit::load(filename, torch::kCPU);
  model.to(device);
  return model;
}

static void remap_device_constants(torch::jit::Block *block,
                                   const torch::Device &device) {
  for (torch::jit::Node *node : block->nodes()) {
    if (node->kind() == torch::jit::prim::Constant &&
        node->outputs().size() == 1 &&
        node->output()->type()->kind() == c10::TypeKind::DeviceObjType &&
        node->hasAttribute(torch::jit::attr::value) &&
        node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::s) {
      node->s_(torch::jit::attr::value, device.str());
    }
    bool has_tensor_input = false;
    for (const torch::jit::Value *input : node->inputs()) {
      has_tensor_input |= input->type()->kind() == c10::TypeKind::TensorType;
    }
    const auto *schema = node->maybeSchema();
    if (!has_tensor_input && schema != nullptr) {
      const auto &arguments = schema->arguments();
      for (size_t i = 0; i < arguments.size() && i < node->inputs().size();
           ++i) {
        const auto input_value = torch::jit::toIValue(node->input(i));
        if (arguments[i].name() == "device" && input_value.has_value() &&
            input_value->isNone()) {
          torch::jit::WithInsertPoint insertion_guard(node);
          auto *device_value =
              node->owningGraph()->insertConstant(torch::jit::IValue(device));
          node->replaceInput(i, device_value);
        }
      }
    }
    for (torch::jit::Block *nested_block : node->blocks()) {
      remap_device_constants(nested_block, device);
    }
  }
}

static void remap_model_device_constants(torch::jit::Module &model,
                                         const torch::Device &device) {
  for (const auto &attribute : model.named_attributes(false)) {
    const auto &value = attribute.value;
    if (value.isTensor()) {
      model.setattr(attribute.name, value.toTensor().to(device));
    } else if (value.isTensorList()) {
      auto tensors = value.toTensorList();
      for (size_t i = 0; i < tensors.size(); ++i) {
        tensors.set(i, tensors.get(i).to(device));
      }
      model.setattr(attribute.name, tensors);
    }
  }
  for (const auto &method : model.get_methods()) {
    remap_device_constants(method.graph()->block(), device);
  }
  for (auto child : model.children()) {
    remap_model_device_constants(child, device);
  }
}

// Collect the device targets hard-coded in a graph (device constants such as
// torch.device("cuda:0") or "cpu").
static void collect_block_device_constants(torch::jit::Block *block,
                                           bool &has_cuda, bool &has_cpu) {
  for (torch::jit::Node *node : block->nodes()) {
    if (node->kind() == torch::jit::prim::Constant &&
        node->outputs().size() == 1 &&
        node->output()->type()->kind() == c10::TypeKind::DeviceObjType &&
        node->hasAttribute(torch::jit::attr::value) &&
        node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::s) {
      const std::string &value = node->s(torch::jit::attr::value);
      if (value.rfind("cuda", 0) == 0) {
        has_cuda = true;
      } else if (value.rfind("cpu", 0) == 0) {
        has_cpu = true;
      }
    }
    for (torch::jit::Block *nested_block : node->blocks()) {
      collect_block_device_constants(nested_block, has_cuda, has_cpu);
    }
  }
}

static void collect_model_device_constants(torch::jit::Module &model,
                                           bool &has_cuda, bool &has_cpu) {
  for (const auto &method : model.get_methods()) {
    collect_block_device_constants(method.graph()->block(), has_cuda, has_cpu);
  }
  for (auto child : model.children()) {
    collect_model_device_constants(child, has_cuda, has_cpu);
  }
}

static void promote_float32_constants(torch::jit::Block *block) {
  for (torch::jit::Node *node : block->nodes()) {
    if (node->kind() == torch::jit::prim::Constant &&
        node->hasAttribute(torch::jit::attr::value) &&
        node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::t) {
      const auto tensor = node->t(torch::jit::attr::value);
      if (tensor.defined() && tensor.scalar_type() == torch::kFloat32) {
        const auto promoted = tensor.to(torch::kFloat64);
        node->t_(torch::jit::attr::value, promoted);
        node->output()->setType(c10::TensorType::create(promoted));
      }
    }
    const auto *schema = node->maybeSchema();
    if (schema != nullptr) {
      const auto &arguments = schema->arguments();
      for (size_t i = 0; i < arguments.size() && i < node->inputs().size();
           ++i) {
        const auto value = torch::jit::toIValue(node->input(i));
        if (arguments[i].name() == "dtype" && value.has_value() &&
            value->isInt() &&
            value->toInt() == static_cast<int64_t>(torch::kFloat32)) {
          // A dtype constant can also be used as a dimension or an index.
          // Replace this edge only; leave other uses and integer tensors
          // intact.
          torch::jit::WithInsertPoint insertion_guard(node);
          auto *dtype = node->owningGraph()->insertConstant(
              static_cast<int64_t>(torch::kFloat64));
          node->replaceInput(i, dtype);
        }
      }
    }
    for (torch::jit::Block *nested : node->blocks()) {
      promote_float32_constants(nested);
    }
  }
}

static void promote_model_float32(torch::jit::Module &model) {
  for (const auto &attribute : model.named_attributes(false)) {
    const auto &value = attribute.value;
    if (value.isTensor() && value.toTensor().defined() &&
        value.toTensor().scalar_type() == torch::kFloat32) {
      model.setattr(attribute.name, value.toTensor().to(torch::kFloat64));
    } else if (value.isTensorList()) {
      auto tensors = value.toTensorList();
      for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors.get(i).defined() &&
            tensors.get(i).scalar_type() == torch::kFloat32) {
          tensors.set(i, tensors.get(i).to(torch::kFloat64));
        }
      }
      model.setattr(attribute.name, tensors);
    }
  }
  for (const auto &method : model.get_methods()) {
    torch::jit::Inline(*method.graph());
    promote_float32_constants(method.graph()->block());
  }
  for (auto child : model.children()) {
    promote_model_float32(child);
  }
}

static void initialize_cuda_linalg(const torch::Device &device) {
  // Warm up each CUDA device once; the active device can change within one
  // process when several GPUs are used.
  static std::mutex mutex;
  static std::set<int> initialized_devices;
  std::lock_guard<std::mutex> lock(mutex);
  if (!initialized_devices.insert(device.index()).second) {
    return;
  }
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  static_cast<void>(at::linalg_eigh(torch::eye(1, options)));
}

/*******************************************************************************
 * \brief Internal helper for creating a Torch tensor from an array.
 * \author Ole Schuett
 ******************************************************************************/
static torch_c_tensor_t *tensor_from_array(const torch::Dtype dtype,
                                           const bool req_grad, const int ndims,
                                           const int64_t sizes[],
                                           void *source) {
  initialize_torch_threads_from_env();
  const auto opts = torch::TensorOptions().dtype(dtype).requires_grad(req_grad);
  const auto sizes_ref = c10::IntArrayRef(sizes, ndims);
  return new torch_c_tensor_t(torch::from_blob(source, sizes_ref, opts));
}

static bool tensor_matches(const torch_c_tensor_t *tensor,
                           const torch::Dtype dtype,
                           const torch::Device &device, const int ndims,
                           const int64_t sizes[]) {
  if (tensor == nullptr || !tensor->defined() ||
      tensor->scalar_type() != dtype || tensor->device() != device ||
      tensor->ndimension() != ndims) {
    return false;
  }
  for (int i = 0; i < ndims; i++) {
    if (tensor->size(i) != sizes[i]) {
      return false;
    }
  }
  return tensor->is_contiguous();
}

static bool reset_tensor_from_array_double(torch_c_tensor_t **tensor,
                                           const bool req_grad, const int ndims,
                                           const int64_t sizes[],
                                           double source[]) {
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  const auto sizes_ref = c10::IntArrayRef(sizes, ndims);
  const bool reallocated =
      !tensor_matches(*tensor, torch::kFloat64, device, ndims, sizes);
  if (reallocated) {
    delete (*tensor);
    const auto opts =
        torch::TensorOptions().dtype(torch::kFloat64).device(device);
    *tensor = new torch_c_tensor_t(torch::empty(sizes_ref, opts).detach());
  }
  const auto source_tensor = torch::from_blob(
      source, sizes_ref, torch::TensorOptions().dtype(torch::kFloat64));
  {
    torch::NoGradGuard no_grad;
    (*tensor)->copy_(source_tensor);
    (*tensor)->mutable_grad() = torch::Tensor();
  }
  (*tensor)->set_requires_grad(req_grad);
  return reallocated;
}

/*******************************************************************************
 * \brief Internal helper for getting the data_ptr and sizes of a Torch tensor.
 * \author Ole Schuett
 ******************************************************************************/
static void *get_data_ptr(const torch_c_tensor_t *tensor,
                          const torch::Dtype dtype, const int ndims,
                          int64_t sizes[]) {
  torch_c_check(tensor->scalar_type() == dtype,
                "Torch tensor has an unexpected dtype");
  torch_c_check(tensor->ndimension() == ndims,
                "Torch tensor has an unexpected number of dimensions");
  for (int i = 0; i < ndims; i++) {
    sizes[i] = tensor->size(i);
  }

  torch_c_check(tensor->is_contiguous(), "Torch tensor is not contiguous");
  return tensor->data_ptr();
};
/*******************************************************************************
 * \brief Copies a model attribute string into a CP2K default-string buffer.
 *        The Fortran side allocates exactly CP2K's default_string_length (80)
 *        bytes, so reject longer strings instead of overflowing.
 ******************************************************************************/
static constexpr size_t torch_c_attr_string_length = 80;

static void copy_attr_string_to_fortran(const std::string &source, char *dest) {
  if (source.size() >= torch_c_attr_string_length) {
    torch_c_api_abort("Torch attribute string exceeds CP2K default length");
  }
  std::memcpy(dest, source.data(), source.size());
}

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * \brief Creates a Torch tensor from an array of int32s.
 *        The passed array has to outlive the tensor!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_from_array_int32(torch_c_tensor_t **tensor,
                                     const bool req_grad, const int ndims,
                                     const int64_t sizes[], int32_t source[]) {
  TORCH_C_TRY
  *tensor = tensor_from_array(torch::kInt32, req_grad, ndims, sizes, source);
  TORCH_C_CATCH("torch_c_tensor_from_array_int32")
}

/*******************************************************************************
 * \brief Creates a Torch tensor from an array of floats.
 *        The passed array has to outlive the tensor!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_from_array_float(torch_c_tensor_t **tensor,
                                     const bool req_grad, const int ndims,
                                     const int64_t sizes[], float source[]) {
  TORCH_C_TRY
  *tensor = tensor_from_array(torch::kFloat32, req_grad, ndims, sizes, source);
  TORCH_C_CATCH("torch_c_tensor_from_array_float")
}

/*******************************************************************************
 * \brief Creates a Torch tensor from an array of int64s.
 *        The passed array has to outlive the tensor!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_from_array_int64(torch_c_tensor_t **tensor,
                                     const bool req_grad, const int ndims,
                                     const int64_t sizes[], int64_t source[]) {
  TORCH_C_TRY
  *tensor = tensor_from_array(torch::kInt64, req_grad, ndims, sizes, source);
  TORCH_C_CATCH("torch_c_tensor_from_array_int64")
}

/*******************************************************************************
 * \brief Creates a Torch tensor from an array of doubles.
 *        The passed array has to outlive the tensor!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_from_array_double(torch_c_tensor_t **tensor,
                                      const bool req_grad, const int ndims,
                                      const int64_t sizes[], double source[]) {
  TORCH_C_TRY
  *tensor = tensor_from_array(torch::kFloat64, req_grad, ndims, sizes, source);
  TORCH_C_CATCH("torch_c_tensor_from_array_double")
}

/*******************************************************************************
 * \brief Releases a string returned from the Torch C API.
 ******************************************************************************/
void torch_c_free_string(char *content) { free(content); }

/*******************************************************************************
 * \brief Reuses or creates a device tensor and reports reallocation.
 ******************************************************************************/
bool torch_c_tensor_reset_from_array_double_status(torch_c_tensor_t **tensor,
                                                   const bool req_grad,
                                                   const int ndims,
                                                   const int64_t sizes[],
                                                   double source[]) {
  TORCH_C_TRY
  return reset_tensor_from_array_double(tensor, req_grad, ndims, sizes, source);
  TORCH_C_CATCH("torch_c_tensor_reset_from_array_double_status")
}

/*******************************************************************************
 * \brief Creates an expanded tensor view along one singleton dimension.
 ******************************************************************************/
void torch_c_tensor_expand_dim(const torch_c_tensor_t *tensor,
                               const int64_t dim, const int64_t size,
                               torch_c_tensor_t **result) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  torch_c_check(*result == NULL, "Torch tensor output must be NULL");
  torch_c_check(dim >= 0, "Torch tensor dimension must be non-negative");
  torch_c_check(dim < tensor->dim(), "Torch tensor dimension is out of range");
  std::vector<int64_t> sizes(tensor->sizes().begin(), tensor->sizes().end());
  torch_c_check(sizes[dim] == 1,
                "Torch tensor expansion requires a singleton dimension");
  sizes[dim] = size;
  *result = new torch_c_tensor_t(tensor->expand(sizes));
  TORCH_C_CATCH("torch_c_tensor_expand_dim")
}

/*******************************************************************************
 * \brief Creates a tensor view narrowed along one dimension.
 ******************************************************************************/
void torch_c_tensor_narrow(const torch_c_tensor_t *tensor, const int64_t dim,
                           const int64_t start_index, const int64_t length,
                           torch_c_tensor_t **result) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  torch_c_check(*result == NULL, "Torch tensor output must be NULL");
  torch_c_check(dim >= 0, "Torch tensor dimension must be non-negative");
  torch_c_check(start_index >= 0,
                "Torch tensor start index must be non-negative");
  torch_c_check(length >= 0, "Torch tensor length must be non-negative");
  torch_c_check(dim < tensor->ndimension(),
                "Torch tensor dimension is out of range");
  torch_c_check(start_index + length <= tensor->size(dim),
                "Torch tensor narrow range is out of bounds");
  *result =
      new torch_c_tensor_t(tensor->narrow(dim, start_index, length).to(device));
  TORCH_C_CATCH("torch_c_tensor_narrow")
}

/*******************************************************************************
 * \brief Returns the data_ptr and sizes of a Torch tensor of int32s.
 *        The returned pointer is only valide during the tensor's live time!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_data_ptr_int32(const torch_c_tensor_t *tensor,
                                   const int ndims, int64_t sizes[],
                                   int32_t **data_ptr) {
  TORCH_C_TRY
  *data_ptr = (int32_t *)get_data_ptr(tensor, torch::kInt32, ndims, sizes);
  TORCH_C_CATCH("torch_c_tensor_data_ptr_int32")
}

/*******************************************************************************
 * \brief Returns the data_ptr and sizes of a Torch tensor of floats.
 *        The returned pointer is only valide during the tensor's lifetime!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_data_ptr_float(const torch_c_tensor_t *tensor,
                                   const int ndims, int64_t sizes[],
                                   float **data_ptr) {
  TORCH_C_TRY
  *data_ptr = (float *)get_data_ptr(tensor, torch::kFloat32, ndims, sizes);
  TORCH_C_CATCH("torch_c_tensor_data_ptr_float")
}

/*******************************************************************************
 * \brief Returns the data_ptr and sizes of a Torch tensor of int64s.
 *        The returned pointer is only valide during the tensor's live time!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_data_ptr_int64(const torch_c_tensor_t *tensor,
                                   const int ndims, int64_t sizes[],
                                   int64_t **data_ptr) {
  TORCH_C_TRY
  *data_ptr = (int64_t *)get_data_ptr(tensor, torch::kInt64, ndims, sizes);
  TORCH_C_CATCH("torch_c_tensor_data_ptr_int64")
}

/*******************************************************************************
 * \brief Returns the data_ptr and sizes of a Torch tensor of doubles.
 *        The returned pointer is only valide during the tensor's live time!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_data_ptr_double(const torch_c_tensor_t *tensor,
                                    const int ndims, int64_t sizes[],
                                    double **data_ptr) {
  TORCH_C_TRY
  *data_ptr = (double *)get_data_ptr(tensor, torch::kFloat64, ndims, sizes);
  TORCH_C_CATCH("torch_c_tensor_data_ptr_double")
}

/*******************************************************************************
 * \brief Runs autograd on a Torch tensor.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_backward(const torch_c_tensor_t *tensor,
                             const torch_c_tensor_t *outer_grad) {
  TORCH_C_TRY
  TorchFloatingPointMaskGuard fpe_guard;
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  tensor->backward(*outer_grad);
  TORCH_C_CATCH("torch_c_tensor_backward")
}

/*******************************************************************************
 * \brief Runs autograd on a scalar Torch tensor.
 ******************************************************************************/
void torch_c_tensor_backward_scalar(const torch_c_tensor_t *tensor) {
  TORCH_C_TRY
  TorchFloatingPointMaskGuard fpe_guard;
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  tensor->backward();
  TORCH_C_CATCH("torch_c_tensor_backward_scalar")
}

/*******************************************************************************
 * \brief Moves a tensor to the active device and makes it an autograd leaf.
 ******************************************************************************/
void torch_c_tensor_to_device_leaf(torch_c_tensor_t **tensor,
                                   const bool req_grad) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  auto moved = (*tensor)->to(device).detach();
  moved.set_requires_grad(req_grad);
  delete (*tensor);
  *tensor = new torch_c_tensor_t(moved);
  TORCH_C_CATCH("torch_c_tensor_to_device_leaf")
}

/*******************************************************************************
 * \brief Select whether Torch wrappers should use CUDA when available.
 ******************************************************************************/
void torch_c_use_cuda(const bool use_cuda) { use_cuda_if_available = use_cuda; }

/*******************************************************************************
 * \brief Returns whether Torch wrappers currently select CUDA when available.
 ******************************************************************************/
bool torch_c_get_use_cuda() { return use_cuda_if_available; }

/*******************************************************************************
 * \brief Returns the gradient of a Torch tensor which was computed by autograd.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_grad(const torch_c_tensor_t *tensor,
                         torch_c_tensor_t **grad) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  const torch::Tensor maybe_grad = tensor->grad();
  torch::Tensor host_grad;
  if (maybe_grad.defined()) {
    host_grad = maybe_grad.detach().cpu().contiguous();
    if (maybe_grad.is_cpu()) {
      host_grad = host_grad.clone();
    }
  } else {
    // A tensor that does not enter the differentiated expression has no
    // gradient; return an explicit zero instead of dereferencing an undefined
    // tensor in release builds.
    host_grad = torch::zeros(
        tensor->sizes(),
        tensor->options().device(torch::kCPU).requires_grad(false));
  }
  *grad = new torch_c_tensor_t(std::move(host_grad));
  TORCH_C_CATCH("torch_c_tensor_grad")
}

/*******************************************************************************
 * \brief Copies three autograd gradients to CPU memory.
 ******************************************************************************/
void torch_c_tensor_grad_batch3(const torch_c_tensor_t *tensor1,
                                const torch_c_tensor_t *tensor2,
                                const torch_c_tensor_t *tensor3,
                                torch_c_tensor_t **grad1,
                                torch_c_tensor_t **grad2,
                                torch_c_tensor_t **grad3) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  torch_c_check(*grad1 == nullptr && *grad2 == nullptr && *grad3 == nullptr,
                "Torch gradient outputs must be NULL");

  const torch::Tensor source1 = tensor1->grad();
  const torch::Tensor source2 = tensor2->grad();
  const torch::Tensor source3 = tensor3->grad();
  const bool all_defined =
      source1.defined() && source2.defined() && source3.defined();
  if (all_defined && use_batched_gradient_readback(source1)) {
    torch_c_check(source1.device() == source2.device() &&
                      source1.device() == source3.device(),
                  "Torch gradients live on different devices");
    auto host1 =
        torch::empty(source1.sizes(),
                     source1.options().device(torch::kCPU).pinned_memory(true));
    auto host2 =
        torch::empty(source2.sizes(),
                     source2.options().device(torch::kCPU).pinned_memory(true));
    auto host3 =
        torch::empty(source3.sizes(),
                     source3.options().device(torch::kCPU).pinned_memory(true));
    host1.copy_(source1, true);
    host2.copy_(source2, true);
    host3.copy_(source3, true);
    torch::cuda::synchronize(device.index());
    *grad1 = new torch_c_tensor_t(std::move(host1));
    *grad2 = new torch_c_tensor_t(std::move(host2));
    *grad3 = new torch_c_tensor_t(std::move(host3));
  } else {
    // Materialize independent host buffers instead of aliasing gradients owned
    // by the autograd graph when they are already contiguous CPU tensors.
    // Tensors without an autograd gradient get an explicit zero buffer.
    const auto materialize = [](const torch::Tensor &source,
                                const torch::Tensor &like) {
      if (!source.defined()) {
        return torch::zeros(like.sizes(), like.options().device(torch::kCPU));
      }
      return source.detach().cpu().contiguous().clone();
    };
    *grad1 = new torch_c_tensor_t(materialize(source1, *tensor1));
    *grad2 = new torch_c_tensor_t(materialize(source2, *tensor2));
    *grad3 = new torch_c_tensor_t(materialize(source3, *tensor3));
  }
  TORCH_C_CATCH("torch_c_tensor_grad_batch3")
}

/*******************************************************************************
 * \brief Releases a Torch tensor and all its ressources.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_tensor_release(torch_c_tensor_t *tensor) { delete (tensor); }

/*******************************************************************************
 * \brief Creates an empty Torch dictionary.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_dict_create(torch_c_dict_t **dict_out) {
  TORCH_C_TRY
  torch_c_check(*dict_out == NULL, "Torch dictionary output must be NULL");
  *dict_out = new c10::Dict<std::string, torch::Tensor>();
  TORCH_C_CATCH("torch_c_dict_create")
}

/*******************************************************************************
 * \brief Clones a Torch dictionary.
 ******************************************************************************/
void torch_c_dict_clone(const torch_c_dict_t *dict, torch_c_dict_t **dict_out) {
  TORCH_C_TRY
  torch_c_check(*dict_out == NULL, "Torch dictionary output must be NULL");
  torch_c_dict_t *clone = new c10::Dict<std::string, torch::Tensor>();
  for (const auto &entry : *dict) {
    clone->insert(entry.key(), entry.value());
  }
  *dict_out = clone;
  TORCH_C_CATCH("torch_c_dict_clone")
}

/*******************************************************************************
 * \brief Inserts a Torch tensor into a Torch dictionary.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_dict_insert(const torch_c_dict_t *dict, const char *key,
                         const torch_c_tensor_t *tensor) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  dict->insert(key, tensor->to(device));
  TORCH_C_CATCH("torch_c_dict_insert")
}

/*******************************************************************************
 * \brief Retrieves a Torch tensor from a Torch dictionary.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_dict_get(const torch_c_dict_t *dict, const char *key,
                      torch_c_tensor_t **tensor) {
  TORCH_C_TRY
  torch_c_check(dict->contains(key), "Torch dictionary key not found");
  *tensor = new torch_c_tensor_t(dict->at(key).cpu().contiguous());
  TORCH_C_CATCH("torch_c_dict_get")
}

/*******************************************************************************
 * \brief Releases a Torch dictionary and all its ressources.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_dict_release(torch_c_dict_t *dict) { delete (dict); }

/*******************************************************************************
 * \brief Loads a Torch model from given "*.pth" file.
 *        In Torch lingo models are called modules.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_load(torch_c_model_t **model_out, const char *filename) {
  TORCH_C_TRY
  torch_c_check(*model_out == NULL, "Torch model output must be NULL");
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  set_jit_fusion_strategy();
  torch::jit::Module *model = new torch::jit::Module();
  *model = load_module_for_device(filename, device);
  model->eval(); // Set inference behavior for modules such as dropout.
  *model_out = model;
  TORCH_C_CATCH("torch_c_model_load")
}

/*******************************************************************************
 * \brief Loads a Torch model and reads two metadata entries.
 ******************************************************************************/
void torch_c_model_load_with_metadata(torch_c_model_t **model_out,
                                      const char *filename, const char *key1,
                                      const char *key2, char **content1,
                                      int *length1, char **content2,
                                      int *length2) {
  TORCH_C_TRY
  torch_c_check(*model_out == NULL, "Torch model output must be NULL");
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  std::unordered_map<std::string, std::string> extra_files = {{key1, ""},
                                                              {key2, ""}};
  set_jit_fusion_strategy();
  torch::jit::Module *model = new torch::jit::Module();
  // Deduplicate Skala constants on CPU without repeated CUDA synchronization.
  *model =
      load_module_for_device(filename, device, &extra_files, device.is_cuda());
  model->eval(); // Set inference behavior for modules such as dropout.
  *model_out = model;
  copy_string_to_c_buffer(extra_files[key1], content1, length1);
  copy_string_to_c_buffer(extra_files[key2], content2, length2);
  TORCH_C_CATCH("torch_c_model_load_with_metadata")
}

/*******************************************************************************
 * \brief Maps serialized TorchScript device constants to the active device.
 ******************************************************************************/
void torch_c_model_remap_device_constants(torch_c_model_t *model) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  // Remap even for CPU so that a CUDA export can run under USE_GPU HOST (and a
  // CPU export under GPU) instead of failing later with a raw device mismatch.
  remap_model_device_constants(*model, device);
  if (device.is_cuda()) {
    initialize_cuda_linalg(device);
  }

  bool has_cuda = false;
  bool has_cpu = false;
  collect_model_device_constants(*model, has_cuda, has_cpu);
  if (device.is_cuda()) {
    torch_c_check(!has_cpu,
                  "SKALA TorchScript model still targets the CPU "
                  "after remapping; use the CUDA export for GPU runs");
  } else {
    torch_c_check(!has_cuda, "SKALA TorchScript model still targets CUDA after "
                             "remapping; use the CPU export or a CUDA build");
  }
  TORCH_C_CATCH("torch_c_model_remap_device_constants")
}

/*******************************************************************************
 * \brief Promote float32 model state and explicit numerical casts to float64.
 *        Trained values are retained exactly; integer/index operations are
 * kept.
 ******************************************************************************/
void torch_c_model_promote_float32(torch_c_model_t *model) {
  TORCH_C_TRY
  TorchFloatingPointMaskGuard fpe_guard;
  torch::NoGradGuard no_grad;
  promote_model_float32(*model);
  TORCH_C_CATCH("torch_c_model_promote_float32")
}

/*******************************************************************************
 * \brief Disables gradients for inference-only model parameters.
 ******************************************************************************/
void torch_c_model_disable_parameter_gradients(torch_c_model_t *model) {
  TORCH_C_TRY
  torch::NoGradGuard no_grad;
  for (auto parameter : model->parameters()) {
    parameter.set_requires_grad(false);
  }
  TORCH_C_CATCH("torch_c_model_disable_parameter_gradients")
}

/*******************************************************************************
 * \brief Evaluates the given Torch model.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_forward(torch_c_model_t *model, const torch_c_dict_t *inputs,
                           torch_c_dict_t *outputs) {
  TORCH_C_TRY

  TorchFloatingPointMaskGuard fpe_guard;
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  auto untyped_output = model->forward({*inputs}).toGenericDict();
  outputs->clear();
  for (const auto &entry : untyped_output) {
    outputs->insert(entry.key().toStringView(), entry.value().toTensor());
  }
  TORCH_C_CATCH("torch_c_model_forward")
}

/*******************************************************************************
 * \brief Evaluates a TorchScript model method expecting argument "mol".
 ******************************************************************************/
void torch_c_model_forward_mol_tensor(torch_c_model_t *model,
                                      const char *method_name,
                                      const torch_c_dict_t *inputs,
                                      torch_c_tensor_t **output) {
  TORCH_C_TRY

  c10::OptionalDeviceGuard guard;
  const auto device = get_device_with_guard(guard);
  // Avoid repeated TorchScript profiling of the frozen Skala model on CUDA.
  std::optional<torch::jit::GraphOptimizerEnabledGuard> graph_optimization;
  if (device.is_cuda()) {
    graph_optimization.emplace(false);
  }
  torch_c_check(*output == NULL, "Torch tensor output must be NULL");
  *output = new torch_c_tensor_t(
      model->get_method(method_name)({*inputs}).toTensor());
  TORCH_C_CATCH("torch_c_model_forward_mol_tensor")
}

/*******************************************************************************
 * \brief Returns the weighted sum of two Torch tensors.
 ******************************************************************************/
void torch_c_tensor_weighted_sum(const torch_c_tensor_t *values,
                                 const torch_c_tensor_t *weights,
                                 torch_c_tensor_t **result) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  const auto weights_on_device = weights->to(values->device());
  *result = new torch_c_tensor_t((*values * weights_on_device).sum());
  TORCH_C_CATCH("torch_c_tensor_weighted_sum")
}

/*******************************************************************************
 * \brief Returns a scalar double value from a Torch tensor.
 ******************************************************************************/
double torch_c_tensor_item_double(const torch_c_tensor_t *tensor) {
  TORCH_C_TRY
  c10::OptionalDeviceGuard guard;
  get_device_with_guard(guard);
  return tensor->item<double>();
  TORCH_C_CATCH("torch_c_tensor_item_double")
}

/*******************************************************************************
 * \brief Releases a Torch model and all its ressources.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_release(torch_c_model_t *model) { delete (model); }

/*******************************************************************************
 * \brief Reads metadata entry from given "*.pth" file.
 *        In Torch lingo they are called extra files.
 *        The returned char array has to be deallocated by caller!
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_read_metadata(const char *filename, const char *key,
                                 char **content, int *length) {
  TORCH_C_TRY

  std::unordered_map<std::string, std::string> extra_files = {{key, ""}};
  torch::jit::load(filename, torch::kCPU, extra_files);
  const std::string &content_str = extra_files[key];
  copy_string_to_c_buffer(content_str, content, length);
  TORCH_C_CATCH("torch_c_model_read_metadata")
}

/*******************************************************************************
 * \brief Returns true iff the Torch CUDA backend is available.
 * \author Ole Schuett
 ******************************************************************************/
bool torch_c_cuda_is_available() { return torch::cuda::is_available(); }

/*******************************************************************************
 * \brief Return the number of CUDA devices visible to Torch.
 ******************************************************************************/
int torch_c_cuda_device_count() {
  TORCH_C_TRY
  return torch::cuda::is_available() ? torch::cuda::device_count() : 0;
  TORCH_C_CATCH("torch_c_cuda_device_count")
}

/*******************************************************************************
 * \brief Set whether to allow TF32.
 *        Needed due to changes in defaults from pytorch 1.7 to 1.11 to >=1.12
 *        See https://pytorch.org/docs/stable/notes/cuda.html
 * \author Gabriele Tocci
 ******************************************************************************/
void torch_c_allow_tf32(const bool allow_tf32) {
  TORCH_C_TRY

  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);
  TORCH_C_CATCH("torch_c_allow_tf32")
}

/******************************************************************************
 * \brief Freeze the Torch model: generic optimization that speeds up model.
 *        See https://pytorch.org/docs/stable/generated/torch.jit.freeze.html
 * \author Gabriele Tocci
 ******************************************************************************/
void torch_c_model_freeze(torch_c_model_t *model) {
  TORCH_C_TRY

  *model = torch::jit::freeze(*model);
  TORCH_C_CATCH("torch_c_model_freeze")
}

/******************************************************************************
 * \brief Freeze a Torch model while preserving one exported method.
 ******************************************************************************/
void torch_c_model_freeze_preserving_method(torch_c_model_t *model,
                                            const char *method_name) {
  TORCH_C_TRY

  const std::vector<std::string> preserved_methods = {method_name};
  torch::jit::freeze_module_inplace(model, preserved_methods);
  TORCH_C_CATCH("torch_c_model_freeze_preserving_method")
}

/*******************************************************************************
 * \brief Retrieves an int64 attribute. Must be called before model freeze.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_get_attr_int64(const torch_c_model_t *model, const char *key,
                                  int64_t *dest) {
  TORCH_C_TRY
  *dest = model->attr(key).toInt();
  TORCH_C_CATCH("torch_c_model_get_attr_int64")
}

/*******************************************************************************
 * \brief Retrieves a double attribute. Must be called before model freeze.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_get_attr_double(const torch_c_model_t *model,
                                   const char *key, double *dest) {
  TORCH_C_TRY
  *dest = model->attr(key).toDouble();
  TORCH_C_CATCH("torch_c_model_get_attr_double")
}

/*******************************************************************************
 * \brief Retrieves a string attribute. Must be called before model freeze.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_get_attr_string(const torch_c_model_t *model,
                                   const char *key, char *dest) {
  TORCH_C_TRY
  copy_attr_string_to_fortran(model->attr(key).toStringRef(), dest);
  TORCH_C_CATCH("torch_c_model_get_attr_string")
}

/*******************************************************************************
 * \brief Retrieves a list attribute's size. Must be called before model freeze.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_get_attr_list_size(const torch_c_model_t *model,
                                      const char *key, int *size) {
  TORCH_C_TRY
  *size = model->attr(key).toList().size();
  TORCH_C_CATCH("torch_c_model_get_attr_list_size")
}

/*******************************************************************************
 * \brief Retrieves a single item from a string list attribute.
 * \author Ole Schuett
 ******************************************************************************/
void torch_c_model_get_attr_strlist(const torch_c_model_t *model,
                                    const char *key, const int index,
                                    char *dest) {
  TORCH_C_TRY
  const auto list = model->attr(key).toList();
  copy_attr_string_to_fortran(list[index].toStringRef(), dest);
  TORCH_C_CATCH("torch_c_model_get_attr_strlist")
}

#ifdef __cplusplus
}
#endif

#endif // defined(__LIBTORCH)

// EOF
