#pragma once

// Based on this tensor implementation: https://github.com/GaoYusong/llm.cpp

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cladtorch {

constexpr size_t kTensorMemAlign = 16;

template <typename T> inline void assert_aligned(T ptr) {
  assert(((uintptr_t)(ptr)) % kTensorMemAlign == 0);
}

enum class TensorType {
  F32,
  I32,
  LEN // number of tensor types
};

constexpr std::array<size_t, 2> kTypeSize = {
    sizeof(float),   // F32
    sizeof(int32_t), // I32
};

template <typename T> inline bool is_type_compatible(TensorType type) {
  switch (type) {
  case TensorType::F32:
    return std::is_same_v<T, float>;
  case TensorType::I32:
    return std::is_same_v<T, int32_t>;
  default:
    throw std::runtime_error("Malformed tensor type");
  }
}

struct Object {
  size_t offset;
  size_t size;
  Object *next;
  std::byte padding[8];
};

constexpr size_t kObjectSize = sizeof(Object);

enum class TensorOp {
  NONE,
  ADD,
  MUL,
  MATMUL,
  LOOKUP,
  NORM,
  BROADCAST,
  VIEW,
  TRANSPOSE,
  GELU,
  SOFTMAX,
  CROSS_ENTROPY,
  LEN // number of tensor operations
};

constexpr std::array<std::string_view, 12> kTensorOpNames = {
    "NONE",      "ADD",  "MUL",       "MATMUL", "LOOKUP",  "NORM",
    "BROADCAST", "VIEW", "TRANSPOSE", "GELU",   "SOFTMAX", "CROSS_ENTROPY"};

constexpr int kMaxTensorDims = 4;
constexpr int kMaxTensorOpParams = 2;

// Forward declaration
class Tensor;

template <typename T> class TensorContextT {
public:
  static constexpr size_t TENSOR_SIZE = sizeof(T);

  explicit TensorContextT(size_t mem_size) : mem_size_(mem_size) {
    mem_buffer_ = new std::byte[mem_size];
    n_objects_ = 0;
    objects_begin_ = nullptr;
    objects_end_ = nullptr;
  }

  ~TensorContextT() { delete[] mem_buffer_; }

  // Prevent copying
  TensorContextT(const TensorContextT &) = delete;
  TensorContextT &operator=(const TensorContextT &) = delete;

  T *new_tensor(const std::vector<int> &dims, float *data) {
    return new_tensor(dims, TensorType::F32,
                      reinterpret_cast<std::byte *>(data));
  }

  T *new_tensor(const std::vector<int> &dims, TensorType type = TensorType::F32,
                std::byte *data = nullptr) {
    const int n_dims = dims.size();

    size_t size_needed = 0;
    if (data == nullptr) {
      size_t data_size = kTypeSize[static_cast<size_t>(type)];
      for (int i = 0; i < n_dims; i++) {
        data_size *= dims[i];
      }
      size_needed += ((data_size + kTensorMemAlign - 1) / kTensorMemAlign) *
                     kTensorMemAlign;
    }
    size_needed += TENSOR_SIZE;

    // layout
    // [Struct Object][Struct Tensor][data]
    std::byte *cur = mem_buffer_;
    if (objects_end_ != nullptr) {
      cur += objects_end_->offset + objects_end_->size;
    }

    if (cur + size_needed + kObjectSize > mem_buffer_ + mem_size_) {
      throw std::runtime_error("Out of tensor memory");
    }

    Object *object = reinterpret_cast<Object *>(cur);

    *object = {.offset = static_cast<size_t>(cur - mem_buffer_) + kObjectSize,
               .size = size_needed,
               .next = nullptr};

    assert_aligned(object);

    if (objects_end_ != nullptr) {
      objects_end_->next = object;
    } else {
      objects_begin_ = object;
    }
    objects_end_ = object;

    T *tensor = reinterpret_cast<T *>(cur + kObjectSize);

    assert_aligned(tensor);

    *tensor = T(this, dims, type,
                data == nullptr ? cur + kObjectSize + TENSOR_SIZE : data);

    assert_aligned(tensor->data_);

    n_objects_++;

    return tensor;
  }

  void print_layout(bool verbose = false) {
    std::cout << "TensorContext Layout" << std::endl;
    std::cout << "---------------------" << std::endl;
    std::cout << "Total memory size: " << mem_size_ << std::endl;
    std::cout << "Used  memory size: "
              << (objects_end_ == nullptr
                      ? 0
                      : (objects_end_->offset + objects_end_->size))
              << std::endl;
    std::cout << "Number of objects: " << n_objects_ << std::endl;
    if (verbose) {
      std::cout << "Objects:" << std::endl;
      Object *cur = objects_begin_;
      while (cur != nullptr) {
        std::cout << "  offset: " << cur->offset << ", size: " << cur->size
                  << std::endl;
        cur = cur->next;
      }
    }
  }

private:
  size_t mem_size_;
  std::byte *mem_buffer_;
  int n_objects_;
  Object *objects_begin_;
  Object *objects_end_;
};

using TensorContext = TensorContextT<Tensor>;

class NormalDist {
public:
  NormalDist() { generator_.seed(std::random_device{}()); }

  float operator()() { return normal_dist_(generator_); }

private:
  std::normal_distribution<float> normal_dist_;
  std::default_random_engine generator_;
};

class Tensor {
public:
  // Add
  Tensor &operator+(Tensor &other) { return operator2(other, TensorOp::ADD); }

  // Mul
  Tensor &operator*(Tensor &other) { return operator2(other, TensorOp::MUL); }

  // Mul by scalar
  Tensor &operator*(float val) {
    assert(type_ == TensorType::F32);
    return *this * *(ctx_->new_tensor({1}, type_)->fill(val));
  }

  // Lookup
  Tensor &operator[](Tensor &index) {
    assert(index.type_ == TensorType::I32);
    assert(n_dims_ + index.n_dims_ - 1 <= kMaxTensorDims);
    std::vector<int> ds;
    for (int i = kMaxTensorDims - 1; i > kMaxTensorDims - n_dims_; i--) {
      ds.push_back(dims_[i]);
    }
    for (int i = kMaxTensorDims - 1; i >= kMaxTensorDims - index.n_dims_; i--) {
      ds.push_back(index.dims_[i]);
    }
    std::reverse(ds.begin(), ds.end());
    Tensor *dst = ctx_->new_tensor(ds, type_);
    dst->op_ = TensorOp::LOOKUP;
    dst->src0_ = this;
    dst->src1_ = &index;
    return *dst;
  }

  // Norm
  Tensor &norm() {
    assert(type_ == TensorType::F32);
    Tensor *dst = ctx_->new_tensor(dims(), type_);
    dst->op_ = TensorOp::NORM;
    dst->src0_ = this;
    return *dst;
  }

  // Gelu
  Tensor &gelu() {
    assert(type_ == TensorType::F32);
    Tensor *dst = ctx_->new_tensor(dims(), type_);
    dst->op_ = TensorOp::GELU;
    dst->src0_ = this;
    return *dst;
  }

  // Softmax
  Tensor &softmax(bool is_casual = false, int vocab_size = 0) {
    assert(type_ == TensorType::F32);
    if (vocab_size > 0) {
      assert(vocab_size <= dims_[kMaxTensorDims - 1]);
    }
    Tensor *dst = ctx_->new_tensor(dims(), type_);
    dst->op_ = TensorOp::SOFTMAX;
    dst->src0_ = this;
    dst->op_params_[0] = is_casual ? 1 : 0;
    dst->op_params_[1] = vocab_size;
    return *dst;
  }

  // CrossEntropy
  Tensor &cross_entropy(Tensor &target) {
    assert(type_ == TensorType::F32 && target.type_ == TensorType::I32);
    auto shape = dims();
    shape.pop_back();
    assert(shape == target.dims());
    Tensor *dst = ctx_->new_tensor(shape, type_);
    dst->op_ = TensorOp::CROSS_ENTROPY;
    dst->src0_ = this;
    dst->src1_ = &target;
    return *dst;
  }

  // Split
  std::vector<Tensor *> split(int size, int axis) {
    assert(axis < n_dims_);
    auto dimi = kMaxTensorDims - n_dims_ + axis;
    assert(dims_[dimi] % size == 0);
    std::vector<Tensor *> tensors;
    if (dims_[dimi] == size) {
      tensors.push_back(this);
      return tensors;
    }

    std::vector<int> shape = dims();
    shape[axis] = size;
    for (int i = 0; i < dims_[dimi] / size; i++) {
      tensors.push_back(&view(shape, i, axis));
    }
    return tensors;
  }

  // View
  Tensor &view(const std::vector<int> &shape, int split_no = 0,
               int split_axis = 0) {
    assert(num_elements() % num_of_elements(shape) == 0);
    int dimi = kMaxTensorDims - n_dims_ + split_axis;
    assert(dims_[dimi] % (num_elements() / num_of_elements(shape)) == 0);
    Tensor *dst = ctx_->new_tensor(shape, type_);
    dst->op_ = TensorOp::VIEW;
    dst->src0_ = this;
    dst->op_params_[0] = split_no;
    dst->op_params_[1] = split_axis;
    return *dst;
  }

  // Transpose
  Tensor &transpose(int axis0, int axis1) {
    assert(axis0 < n_dims_ && axis1 < n_dims_);
    auto dimi0 = kMaxTensorDims - n_dims_ + axis0;
    auto dimi1 = kMaxTensorDims - n_dims_ + axis1;
    std::vector<int> shape = dims();
    std::swap(shape[axis0], shape[axis1]);
    Tensor *dst = ctx_->new_tensor(shape, type_);
    dst->op_ = TensorOp::TRANSPOSE;
    dst->src0_ = this;
    dst->op_params_[0] = dimi0;
    dst->op_params_[1] = dimi1;
    return *dst;
  }

  // Matmul
  // (B, M, N) x (B, P, N) -> (B, M, P)
  // we assume that the input tensors are in the format (B, M, N) and (B, P, N)
  Tensor &matmul(Tensor &other_ref) {
    auto other = &other_ref;
    assert(other != this);
    if (!can_matmul(*this, *other)) {
      assert(can_broadcast_to(*other, *this, 2));
      other = &broadcast_to(ctx_, *other, *this, 2);
      assert(can_matmul(*this, *other));
    }
    std::vector<int> dst_dims = {dims_[0], dims_[1], dims_[2], other->dims_[2]};
    dst_dims.erase(dst_dims.begin(),
                   dst_dims.begin() + dst_dims.size() - n_dims_);

    Tensor *dst = ctx_->new_tensor(dst_dims);
    dst->op_ = TensorOp::MATMUL;
    dst->src0_ = this;
    dst->src1_ = other;
    return *dst;
  }

  void forward() {
    std::vector<Tensor *> sorted = topo_sort(this);

    for (auto *t : sorted) {
      switch (t->op_) {
      case TensorOp::ADD:
        add_forward(t, t->src0_, t->src1_);
        break;
      case TensorOp::MUL:
        mul_forward(t, t->src0_, t->src1_);
        break;
      case TensorOp::MATMUL:
        matmul_forward(t, t->src0_, t->src1_);
        break;
      case TensorOp::LOOKUP:
        lookup_forward(t, t->src0_, t->src1_);
        break;
      case TensorOp::NORM:
        norm_forward(t, t->src0_);
        break;
      case TensorOp::TRANSPOSE:
        transpose_forward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
        break;
      case TensorOp::VIEW:
        view_forward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
        break;
      case TensorOp::BROADCAST:
        broadcast_forward(t, t->src0_);
        break;
      case TensorOp::GELU:
        gelu_forward(t, t->src0_);
        break;
      case TensorOp::SOFTMAX:
        softmax_forward(t, t->src0_, t->op_params_[0] != 0, t->op_params_[1]);
        break;
      case TensorOp::CROSS_ENTROPY:
        cross_entropy_forward(t, t->src0_, t->src1_);
        break;
      case TensorOp::NONE:
        // no-op
        break;
      default:
        throw std::runtime_error(
            "Forward(): Not implemented, " +
            std::string(kTensorOpNames[static_cast<size_t>(t->op_)]));
      }
    }
  }

  void backward(bool init_grad = true, float init_val = 1.0f) {
    std::vector<Tensor *> sorted = topo_sort(this);

    if (init_grad) {
      alloc_grad(false)->grad()->fill(init_val);
    }

    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
      Tensor *t = *it;
      switch (t->op_) {
      case TensorOp::ADD:
        add_backward(t, t->src0_, t->src1_);
        break;
      case TensorOp::MUL:
        mul_backward(t, t->src0_, t->src1_);
        break;
      case TensorOp::MATMUL:
        matmul_backward(t, t->src0_, t->src1_);
        break;
      case TensorOp::LOOKUP:
        lookup_backward(t, t->src0_, t->src1_);
        break;
      case TensorOp::NORM:
        norm_backward(t, t->src0_);
        break;
      case TensorOp::TRANSPOSE:
        transpose_backward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
        break;
      case TensorOp::VIEW:
        view_backward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
        break;
      case TensorOp::BROADCAST:
        broadcast_backward(t, t->src0_);
        break;
      case TensorOp::GELU:
        gelu_backward(t, t->src0_);
        break;
      case TensorOp::SOFTMAX:
        softmax_backward(t, t->src0_, t->op_params_[0] != 0, t->op_params_[1]);
        break;
      case TensorOp::CROSS_ENTROPY:
        cross_entropy_backward(t, t->src0_, t->src1_);
        break;
      case TensorOp::NONE:
        // no-op
        break;
      default:
        throw std::runtime_error(
            "Backward(): Not implemented, " +
            std::string(kTensorOpNames[static_cast<size_t>(t->op_)]));
      }
    }
  }

  void zero_grad() {
    std::vector<Tensor *> sorted = topo_sort(this);

    for (auto t : sorted) {
      if (t->grad_ != nullptr) {
        t->grad_->fill(0.0f);
      }
    }
  }

  void print_tensor(bool include_data = true, size_t sample_size = 10) {
    std::cout << "Tensor" << std::endl;
    std::cout << "------" << std::endl;
    std::cout << "n_dims: " << n_dims_ << std::endl;
    std::cout << "dims: ";
    for (int i = kMaxTensorDims - n_dims_; i < kMaxTensorDims; i++) {
      std::cout << dims_[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "stride: ";
    for (int i = kMaxTensorDims - n_dims_; i < kMaxTensorDims; i++) {
      std::cout << strides_[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "op: " << kTensorOpNames[static_cast<size_t>(op_)] << "("
              << this << ")" << std::endl;
    if (src0_ != nullptr) {
      std::cout << "src0: " << kTensorOpNames[static_cast<size_t>(src0_->op_)]
                << "(" << src0_ << ")" << std::endl;
    }
    if (src1_ != nullptr) {
      std::cout << "src1: " << kTensorOpNames[static_cast<size_t>(src1_->op_)]
                << "(" << src1_ << ")" << std::endl;
    }

    if (include_data) {
      std::cout << "data: \n";
      size_t upto = std::min(n_vec(), sample_size);

      for (size_t i = 0; i < upto; i++) {
        vec_print(vsize(), type_,
                  data_ +
                      i * vstride() * kTypeSize[static_cast<size_t>(type_)]);
        std::cout << std::endl;
      }

      if (grad_ != nullptr) {
        std::cout << "grad: \n";
        for (size_t i = 0; i < upto; i++) {
          vec_print(grad_->vsize(), type_,
                    grad_->data_ + i * grad_->vstride() *
                                       kTypeSize[static_cast<size_t>(type_)]);
          std::cout << std::endl;
        }
      }
    }
  }

  TensorType type() const { return type_; }

  std::byte *data() { return data_; }

  Tensor *grad() { return grad_; }

  // Just for testing
  Tensor *random_grad() {
    grad_ = ctx_->new_tensor(dims())->random_norm();
    return this;
  }

  // Just for testing
  Tensor *fill_grad(float *data) {
    assert(grad_ == nullptr);
    grad_ = ctx_->new_tensor(dims())->fill(data);
    return this;
  }

  Tensor *alloc_grad(bool init = true) {
    if (grad_ == nullptr) {
      grad_ = ctx_->new_tensor(dims());
      if (init) {
        grad_->fill(0.0f);
      }
    }
    return this;
  }

  Tensor *copy_data_from(const Tensor &other) {
    assert(same_shape(other));
    std::memcpy(data_, other.data_,
                num_elements() * kTypeSize[static_cast<size_t>(type_)]);
    return this;
  }

  std::vector<Tensor *> tensors() { return topo_sort(this); }

  std::vector<float> flatten() const { return flatten<float>(); }

  template <typename T> std::vector<T> flatten() const {
    assert(is_type_compatible<T>(type_));
    assert(is_contiguous());
    if (data_ == nullptr) {
      return {};
    }
    T *ptr = reinterpret_cast<T *>(data_);
    std::vector<T> vec(ptr, ptr + num_elements());
    return vec;
  }

  size_t num_elements() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return static_cast<size_t>(dims_[0]) * dims_[1] * dims_[2] * dims_[3];
  }

  template <typename T> Tensor *fill(const std::vector<T> &in_data) {
    assert(in_data.size() == num_elements());
    return fill(in_data.data());
  }

  template <typename T>
  typename std::enable_if<!std::is_pointer<T>::value, Tensor *>::type
  fill(T val) {
    assert(is_type_compatible<T>(type_));
    for (size_t i = 0; i < n_vec(); i++) {
      vec_fill(vsize(), reinterpret_cast<T *>(data_) + i * vstride(), val);
    }
    return this;
  }

  template <typename T>
  typename std::enable_if<std::is_scalar<T>::value, Tensor *>::type
  fill(const T *in_data) {
    assert(is_type_compatible<T>(type_));
    assert(is_contiguous());
    for (size_t i = 0; i < n_vec(); i++) {
      vec_fill(vsize(), reinterpret_cast<T *>(data_) + i * vstride(),
               in_data + i * vstride());
    }
    return this;
  }

  Tensor *random_norm() {
    assert(type_ == TensorType::F32);
    assert(is_contiguous());
    for (size_t i = 0; i < n_vec(); i++) {
      vec_random_norm(vsize(),
                      reinterpret_cast<float *>(data_) + i * vstride());
    }
    return this;
  }

  std::vector<int> dims() const {
    return std::vector<int>(dims_ + kMaxTensorDims - n_dims_,
                            dims_ + kMaxTensorDims);
  }

  std::vector<size_t> strides() const {
    return std::vector<size_t>(strides_ + kMaxTensorDims - n_dims_,
                               strides_ + kMaxTensorDims);
  }

  bool is_contiguous() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return strides_[3] == 1 && strides_[2] == strides_[3] * dims_[3] &&
           strides_[1] == strides_[2] * dims_[2] &&
           strides_[0] == strides_[1] * dims_[1];
  }

  bool same_shape(const Tensor &other, bool check_type = true,
                  bool check_stride = false) const {
    return dims() == other.dims() &&
           (!check_stride || strides() == other.strides()) &&
           (!check_type || type_ == other.type_);
  }

private:
  Tensor() = delete;

  Tensor(TensorContextT<Tensor> *ctx, const std::vector<int> &shape,
         TensorType type, std::byte *data)
      : ctx_(ctx), n_dims_(shape.size()), data_(data), type_(type),
        op_(TensorOp::NONE), grad_(nullptr), src0_(nullptr), src1_(nullptr) {
    assert(n_dims_ <= kMaxTensorDims);

    for (int i = 0; i < n_dims_; i++) {
      dims_[i + kMaxTensorDims - n_dims_] = shape[i];
    }
    for (int i = 0; i < kMaxTensorDims - n_dims_; i++) {
      dims_[i] = 1;
    }
    strides_[kMaxTensorDims - 1] = 1;
    for (int i = kMaxTensorDims - 2; i >= 0; i--) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }

  Tensor &operator2(Tensor &other_ref, TensorOp op) {
    auto other = &other_ref;
    assert(other != this);
    if (!same_shape(*other)) {
      assert(can_broadcast_to(*other, *this));
      other = &broadcast_to(ctx_, *other, *this);
    }
    Tensor *dst = ctx_->new_tensor(dims());
    dst->op_ = op;
    dst->src0_ = this;
    dst->src1_ = other;
    return *dst;
  }

  static bool can_matmul(const Tensor &src0, const Tensor &src1) {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return src0.n_dims_ >= 2 && src0.n_dims_ == src1.n_dims_ &&
           src0.dims_[3] == src1.dims_[3] && src0.dims_[0] == src1.dims_[0] &&
           src0.dims_[1] == src1.dims_[1];
  }

  // start_dim_r is the starting dimension from the right
  static bool can_broadcast_to(const Tensor &from, const Tensor &to,
                               int start_dim_r = 0) {
    const auto &shape = to.dims();
    bool ok = shape.size() >= from.n_dims_ && shape.size() <= kMaxTensorDims;
    assert(from.n_dims_ >= start_dim_r);
    for (int i = start_dim_r; i < from.n_dims_; i++) {
      ok = ok &&
           (from.dims_[kMaxTensorDims - i - 1] == shape[shape.size() - i - 1] ||
            from.dims_[kMaxTensorDims - i - 1] == 1);
    }
    return ok;
  }

  // start_dim_r is the starting dimension from the right
  static Tensor &broadcast_to(TensorContext *ctx, Tensor &from,
                              const Tensor &to, int start_dim_r = 0) {
    // check that the shape is compatible with the current tensor
    assert(can_broadcast_to(from, to, start_dim_r));
    auto dshape = to.dims();
    for (int i = 0; i < start_dim_r; i++) {
      dshape[dshape.size() - i - 1] = from.dims_[kMaxTensorDims - i - 1];
    }
    Tensor *dst = ctx->new_tensor(dshape, from.type_);
    dst->op_ = TensorOp::BROADCAST;
    dst->src0_ = &from;
    return *dst;
  }

  size_t n_vec() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return static_cast<size_t>(dims_[0]) * dims_[1] * dims_[2];
  }

  size_t vstride() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return strides_[2];
  }

  size_t vsize() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return static_cast<size_t>(dims_[3]);
  }

  size_t n_mat() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return static_cast<size_t>(dims_[0]) * dims_[1];
  }

  std::tuple<int, int> mat() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return {dims_[2], dims_[3]};
  }

  size_t mstride() const {
    static_assert(kMaxTensorDims == 4,
                  "MAX_TENSOR_DIMS is not 4 - update this function");
    return strides_[1];
  }

  static size_t num_of_elements(const std::vector<int> &shape) {
    size_t e = 1;
    for (auto s : shape) {
      e *= s;
    }
    return e;
  }

  // Add
  static void add_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
    assert(dst->type_ == TensorType::F32);
    assert(src0->same_shape(*src1) && src1->same_shape(*dst));
    assert(src0->is_contiguous() && src1->is_contiguous() &&
           dst->is_contiguous());

    size_t n = dst->n_vec();
    for (size_t i = 0; i < n; i++) {
      vec_add(dst->vsize(),
              reinterpret_cast<float *>(dst->data_) + i * dst->vstride(),
              reinterpret_cast<float *>(src0->data_) + i * src0->vstride(),
              reinterpret_cast<float *>(src1->data_) + i * src1->vstride());
    }
  }

  static void add_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
    if (src0->grad_ == nullptr) {
      src0->alloc_grad(false)->grad()->copy_data_from(*dst->grad_);
    } else {
      add_forward(src0->grad_, src0->grad_, dst->grad_);
    }
    if (src1->grad_ == nullptr) {
      src1->alloc_grad(false)->grad()->copy_data_from(*dst->grad_);
    } else {
      add_forward(src1->grad_, src1->grad_, dst->grad_);
    }
  }

  // Mul
  static void mul_forward(Tensor *dst, Tensor *src0, Tensor *src1,
                          bool is_acc = false) {
    assert(dst->type_ == TensorType::F32);
    assert(src0->same_shape(*src1) && src1->same_shape(*dst));
    assert(src0->is_contiguous() && src1->is_contiguous() &&
           dst->is_contiguous());

    size_t n = dst->n_vec(), m = dst->vsize();
    if (!is_acc) {
      for (size_t i = 0; i < n; i++) {
        float *out = reinterpret_cast<float *>(dst->data_) + i * dst->vstride();
        float *in0 =
            reinterpret_cast<float *>(src0->data_) + i * src0->vstride();
        float *in1 =
            reinterpret_cast<float *>(src1->data_) + i * src1->vstride();
        for (size_t j = 0; j < m; j++) {
          out[j] = in0[j] * in1[j];
        }
      }
    } else {
      for (size_t i = 0; i < n; i++) {
        float *out = reinterpret_cast<float *>(dst->data_) + i * dst->vstride();
        float *in0 =
            reinterpret_cast<float *>(src0->data_) + i * src0->vstride();
        float *in1 =
            reinterpret_cast<float *>(src1->data_) + i * src1->vstride();
        for (size_t j = 0; j < m; j++) {
          out[j] += in0[j] * in1[j];
        }
      }
    }
  }

  static void mul_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
    src0->alloc_grad();
    mul_forward(src0->grad_, dst->grad_, src1, true);

    src1->alloc_grad();
    mul_forward(src1->grad_, dst->grad_, src0, true);
  }

  // Matmul
  static void matmul_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
    assert(src0->n_mat() == dst->n_mat() && src1->n_mat() == dst->n_mat());
    assert(dst->type_ == TensorType::F32 && src0->type_ == dst->type_ &&
           src1->type_ == dst->type_);
    assert(src0->is_contiguous() && src1->is_contiguous() &&
           dst->is_contiguous());

    size_t n = dst->dims_[2], m = dst->dims_[3], p = src0->dims_[3];
#pragma omp parallel for collapse(2)
    for (size_t mati = 0; mati < dst->n_mat(); mati++) {
      for (size_t i = 0; i < n; i++) {
        float *out = reinterpret_cast<float *>(dst->data_) +
                     mati * dst->mstride() + i * dst->strides_[2];
        float *in0 = reinterpret_cast<float *>(src0->data_) +
                     mati * src0->mstride() + i * src0->strides_[2];
        for (size_t j = 0; j < m; j++) {
          float *in1 = reinterpret_cast<float *>(src1->data_) +
                       mati * src1->mstride() + j * src1->strides_[2];
          out[j] = vec_dot_f32(p, in0, in1);
        }
      }
    }
  }

  static void matmul_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
    src0->alloc_grad();
    src1->alloc_grad();

    size_t matn = dst->n_mat();
    float *dout = reinterpret_cast<float *>(dst->grad_->data_);
    float *din0 = reinterpret_cast<float *>(src0->grad_->data_);
    float *in0 = reinterpret_cast<float *>(src0->data_);
    float *din1 = reinterpret_cast<float *>(src1->grad_->data_);
    float *in1 = reinterpret_cast<float *>(src1->data_);

    // src0->grad += dst->grad matmul src1^T
    size_t n = src0->dims_[2], m = src0->dims_[3], p = dst->dims_[3];
#pragma omp parallel for collapse(2)
    for (size_t mati = 0; mati < matn; mati++) {
      float *in1_ma = in1 + mati * src1->mstride();
      for (size_t i = 0; i < n; i++) {
        float *din0_mai = din0 + mati * src0->mstride() + i * src0->strides_[2];
        float *dout_mai = dout + mati * dst->mstride() + i * dst->strides_[2];
        for (size_t k = 0; k < p; k++) {
          for (size_t j = 0; j < m; j++) {
            din0_mai[j] += dout_mai[k] * in1_ma[k * src1->strides_[2] + j];
          }
        }
      }
    }

    // src1->grad += dst->grad^T matmul src0^T
    n = src1->dims_[2], m = src1->dims_[3], p = dst->dims_[2];
#pragma omp parallel for
    for (size_t mati = 0; mati < matn; mati++) {
      float *dout_ma = dout + mati * dst->mstride();
      float *in0_ma = in0 + mati * src0->mstride();
      for (size_t k = 0; k < p; k++) {
        for (size_t i = 0; i < n; i++) {
          float *din1_mai =
              din1 + mati * src1->mstride() + i * src1->strides_[2];
          for (size_t j = 0; j < m; j++) {
            din1_mai[j] += dout_ma[k * dst->strides_[2] + i] *
                           in0_ma[k * src0->strides_[2] + j];
          }
        }
      }
    }
  }

  static float vec_dot_f32(const size_t n, const float *va, const float *vb) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
      sum += va[i] * vb[i];
    }
    return sum;
  }

  // Lookup
  static void lookup_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
    assert(dst->type_ == src0->type_ && src1->type_ == TensorType::I32);
    assert(src0->is_contiguous() && src1->is_contiguous() &&
           dst->is_contiguous());

    size_t i0_size = src0->dims_[kMaxTensorDims - src0->n_dims_];
    size_t i0_stride = src0->strides_[kMaxTensorDims - src0->n_dims_];
    size_t type_size = kTypeSize[static_cast<size_t>(src0->type_)];

    for (size_t i = 0; i < src1->num_elements(); i++) {
      int32_t idx = reinterpret_cast<int32_t *>(src1->data_)[i];
      assert(idx >= 0 && idx < i0_size);
      std::memcpy(dst->data_ + i * i0_stride * type_size,
                  src0->data_ + idx * i0_stride * type_size,
                  i0_stride * type_size);
    }
  }

  static void lookup_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
    src0->alloc_grad();

    size_t i0_stride = src0->strides_[kMaxTensorDims - src0->n_dims_];
    size_t type_size = kTypeSize[static_cast<size_t>(src0->type_)];

    for (size_t i = 0; i < src1->num_elements(); i++) {
      int32_t idx = reinterpret_cast<int32_t *>(src1->data_)[i];
      vec_add(i0_stride,
              reinterpret_cast<float *>(src0->grad_->data_) + idx * i0_stride,
              reinterpret_cast<float *>(src0->grad_->data_) + idx * i0_stride,
              reinterpret_cast<float *>(dst->grad_->data_) + i * i0_stride);
    }
  }

  // Norm
  static void norm_forward(Tensor *dst, Tensor *src) {
    assert(src->type_ == TensorType::F32 && dst->type_ == src->type_);
    assert(src->is_contiguous() && dst->is_contiguous());

    for (size_t idx = 0; idx < src->n_vec(); idx++) {
      const float *vec =
          reinterpret_cast<float *>(src->data_) + idx * src->vstride();
      size_t vec_size = src->vsize();

      // calculate the mean and the rstd (without bias correction)
      float mean = vec_mean(vec_size, vec);
      float rstd = vec_rstd(vec_size, vec, mean);

      float *out = reinterpret_cast<float *>(dst->data_) + idx * dst->vstride();
      for (size_t i = 0; i < vec_size; i++) {
        out[i] = (vec[i] - mean) * rstd;
      }
    }
  }

  static void norm_backward(Tensor *dst, Tensor *src) {
    src->alloc_grad();

    for (size_t idx = 0; idx < src->n_vec(); idx++) {
      const float *a =
          reinterpret_cast<float *>(src->data_) + idx * src->vstride();
      const float *b =
          reinterpret_cast<float *>(dst->data_) + idx * dst->vstride();
      size_t vec_size = src->vsize();
      assert(vec_size > 0);

      float mean = vec_mean(vec_size, a);
      float rstd = vec_rstd(vec_size, a, mean);

      float *sgrad =
          reinterpret_cast<float *>(src->grad_->data_) + idx * src->vstride();
      float *dgrad =
          reinterpret_cast<float *>(dst->grad_->data_) + idx * dst->vstride();

      float dgrad_mean = 0.0f, dgrad2_mean = 0.0f;
      for (size_t i = 0; i < vec_size; i++) {
        dgrad_mean += dgrad[i];
        dgrad2_mean += dgrad[i] * b[i];
      }
      dgrad_mean /= vec_size;
      dgrad2_mean /= vec_size;

      for (size_t i = 0; i < vec_size; i++) {
        sgrad[i] += ((dgrad[i] - dgrad_mean) - dgrad2_mean * b[i]) * rstd;
      }
    }
  }

  static float vec_mean(size_t vec_size, const float *src) {
    float sum = 0.0f;
    for (size_t i = 0; i < vec_size; i++) {
      sum += src[i];
    }
    return sum / vec_size;
  }

  static float vec_rstd(size_t vec_size, const float *src, float mean) {
    constexpr float eps = 1e-5f;
    float sum = 0.0f;
    for (size_t i = 0; i < vec_size; i++) {
      float diff = src[i] - mean;
      sum += diff * diff;
    }
    float var = sum / vec_size;
    return 1.0f / std::sqrt(var + eps);
  }

  // View
  static void view_forward(Tensor *dst, Tensor *src, int split_no,
                           int split_axis) {
    assert(dst->type_ == TensorType::F32);
    int dimi, offset;
    int sdims[kMaxTensorDims];
    calculate_split(sdims, dimi, offset, dst, src, split_no, split_axis);

    size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2],
           d3 = dst->dims_[3];
    for (size_t i0 = 0; i0 < d0; i0++) {
      for (size_t i1 = 0; i1 < d1; i1++) {
        for (size_t i2 = 0; i2 < d2; i2++) {
          for (size_t i3 = 0; i3 < d3; i3++) {
            size_t idx = i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                         i2 * dst->strides_[2] + i3 * dst->strides_[3];
            float *dd = reinterpret_cast<float *>(dst->data_) + idx;
            size_t sidx[4] = {idx / sdims[1] / sdims[2] / sdims[3],
                              idx / sdims[2] / sdims[3] % sdims[1],
                              idx / sdims[3] % sdims[2], idx % sdims[3]};
            sidx[dimi] += offset;
            float *sd = reinterpret_cast<float *>(src->data_) +
                        sidx[0] * src->strides_[0] +
                        sidx[1] * src->strides_[1] +
                        sidx[2] * src->strides_[2] + sidx[3] * src->strides_[3];
            *dd = *sd;
          }
        }
      }
    }
  }

  static void view_backward(Tensor *dst, Tensor *src, int split_no,
                            int split_axis) {
    src->alloc_grad();

    int dimi, offset;
    int sdims[kMaxTensorDims];
    calculate_split(sdims, dimi, offset, dst, src, split_no, split_axis);

    size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2],
           d3 = dst->dims_[3];
    for (size_t i0 = 0; i0 < d0; i0++) {
      for (size_t i1 = 0; i1 < d1; i1++) {
        for (size_t i2 = 0; i2 < d2; i2++) {
          for (size_t i3 = 0; i3 < d3; i3++) {
            size_t idx = i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                         i2 * dst->strides_[2] + i3 * dst->strides_[3];
            float *dd = reinterpret_cast<float *>(dst->grad_->data_) + idx;
            size_t sidx[4] = {idx / sdims[1] / sdims[2] / sdims[3],
                              idx / sdims[2] / sdims[3] % sdims[1],
                              idx / sdims[3] % sdims[2], idx % sdims[3]};
            sidx[dimi] += offset;
            float *sd = reinterpret_cast<float *>(src->grad_->data_) +
                        sidx[0] * src->strides_[0] +
                        sidx[1] * src->strides_[1] +
                        sidx[2] * src->strides_[2] + sidx[3] * src->strides_[3];
            *sd += *dd;
          }
        }
      }
    }
  }

  static void calculate_split(int *dims, int &dimi, int &offset, Tensor *dst,
                              Tensor *src, int split_no, int split_axis) {
    dimi = kMaxTensorDims - src->n_dims_ + split_axis;
    int split_size =
        src->dims_[dimi] / (src->num_elements() / dst->num_elements());
    offset = split_no * split_size;

    for (int i = 0; i < kMaxTensorDims; i++) {
      dims[i] = src->dims_[i];
    }
    dims[dimi] = split_size;
  }

  // Transpose
  static void transpose_forward(Tensor *dst, Tensor *src, int dimi0,
                                int dimi1) {
    assert(dst->type_ == TensorType::F32);

    transpose_impl(reinterpret_cast<float *>(dst->data_), dst->strides_,
                   reinterpret_cast<float *>(src->data_), src->strides_,
                   src->dims_, dimi0, dimi1, false);
  }

  static void transpose_backward(Tensor *dst, Tensor *src, int dimi0,
                                 int dimi1) {
    src->alloc_grad();

    transpose_impl(reinterpret_cast<float *>(src->grad_->data_), src->strides_,
                   reinterpret_cast<float *>(dst->grad_->data_), dst->strides_,
                   dst->dims_, dimi0, dimi1, true);
  }

  static void transpose_impl(float *out, size_t *out_strides, float *in,
                             size_t *in_strides, int *dims, int dimi0,
                             int dimi1, bool is_acc) {
    size_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
    for (size_t i0 = 0; i0 < d0; i0++) {
      for (size_t i1 = 0; i1 < d1; i1++) {
        for (size_t i2 = 0; i2 < d2; i2++) {
          for (size_t i3 = 0; i3 < d3; i3++) {
            float *sd = in + (i0 * in_strides[0] + i1 * in_strides[1] +
                              i2 * in_strides[2] + i3 * in_strides[3]);
            size_t di[4] = {i0, i1, i2, i3};
            std::swap(di[dimi0], di[dimi1]);
            float *dd = out + (di[0] * out_strides[0] + di[1] * out_strides[1] +
                               di[2] * out_strides[2] + di[3] * out_strides[3]);

            if (is_acc) {
              *dd += *sd;
            } else {
              *dd = *sd;
            }
          }
        }
      }
    }
  }

  // Broadcast
  static void broadcast_forward(Tensor *dst, Tensor *src0) {
    assert(dst->type_ == TensorType::F32);
    size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2],
           d3 = dst->dims_[3];

    for (size_t i0 = 0; i0 < d0; i0++) {
      for (size_t i1 = 0; i1 < d1; i1++) {
        for (size_t i2 = 0; i2 < d2; i2++) {
          for (size_t i3 = 0; i3 < d3; i3++) {
            float *dd = reinterpret_cast<float *>(dst->data_) +
                        (i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                         i2 * dst->strides_[2] + i3 * dst->strides_[3]);
            size_t si0 = i0 % src0->dims_[0], si1 = i1 % src0->dims_[1],
                   si2 = i2 % src0->dims_[2], si3 = i3 % src0->dims_[3];
            float *sd = reinterpret_cast<float *>(src0->data_) +
                        (si0 * src0->strides_[0] + si1 * src0->strides_[1] +
                         si2 * src0->strides_[2] + si3 * src0->strides_[3]);
            *dd = *sd;
          }
        }
      }
    }
  }

  static void broadcast_backward(Tensor *dst, Tensor *src0) {
    src0->alloc_grad();

    size_t d0 = dst->grad_->dims_[0], d1 = dst->grad_->dims_[1],
           d2 = dst->grad_->dims_[2], d3 = dst->grad_->dims_[3];

    for (size_t i0 = 0; i0 < d0; i0++) {
      for (size_t i1 = 0; i1 < d1; i1++) {
        for (size_t i2 = 0; i2 < d2; i2++) {
          for (size_t i3 = 0; i3 < d3; i3++) {
            auto dd =
                reinterpret_cast<float *>(dst->grad_->data_) +
                (i0 * dst->grad_->strides_[0] + i1 * dst->grad_->strides_[1] +
                 i2 * dst->grad_->strides_[2] + i3 * dst->grad_->strides_[3]);
            size_t si0 = i0 % src0->grad_->dims_[0],
                   si1 = i1 % src0->grad_->dims_[1],
                   si2 = i2 % src0->grad_->dims_[2],
                   si3 = i3 % src0->grad_->dims_[3];
            auto sd = reinterpret_cast<float *>(src0->grad_->data_) +
                      (si0 * src0->grad_->strides_[0] +
                       si1 * src0->grad_->strides_[1] +
                       si2 * src0->grad_->strides_[2] +
                       si3 * src0->grad_->strides_[3]);
            *sd += *dd;
          }
        }
      }
    }
  }

  // Gelu
  static void gelu_forward(Tensor *dst, Tensor *src) {
    assert(dst->same_shape(*src, true, true));

    auto out = reinterpret_cast<float *>(dst->data_);
    auto inp = reinterpret_cast<float *>(src->data_);
    auto N = dst->num_elements();
    float s = std::sqrt(2.0f / M_PI);
    for (int i = 0; i < N; i++) {
      float x = inp[i];
      float cube = 0.044715f * x * x * x;
      out[i] = 0.5f * x * (1.0f + std::tanh(s * (x + cube)));
    }
  }

  static void gelu_backward(Tensor *dst, Tensor *src) {
    src->alloc_grad();

    auto N = dst->num_elements();
    auto dinp = reinterpret_cast<float *>(src->grad_->data_);
    auto inp = reinterpret_cast<float *>(src->data_);
    auto dout = reinterpret_cast<float *>(dst->grad_->data_);

    float s = std::sqrt(2.0f / M_PI);
    for (int i = 0; i < N; i++) {
      float x = inp[i];
      float cube = 0.044715f * x * x * x;
      float tanh_arg = s * (x + cube);
      float tanh_out = std::tanh(tanh_arg);
      float cosh_out = std::cosh(tanh_arg);
      float sech_out = 1.0f / (cosh_out * cosh_out);
      float local_grad =
          0.5f * (1.0f + tanh_out) +
          x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
      dinp[i] += local_grad * dout[i];
    }
  }

  // Softmax
  static void softmax_forward(Tensor *dst, Tensor *src, bool is_casual,
                              int vocab_size) {
    assert(dst->same_shape(*src));
    assert(dst->is_contiguous() && src->is_contiguous());

    auto [n, m] = dst->mat();
    assert(m > 0);

    for (size_t mati = 0; mati < dst->n_mat(); mati++) {
      for (size_t i = 0; i < n; i++) {
        auto logits = reinterpret_cast<float *>(src->data_) +
                      mati * src->mstride() + i * m;
        auto probs = reinterpret_cast<float *>(dst->data_) +
                     mati * dst->mstride() + i * m;
        int V = vocab_size > 0 ? vocab_size : m;
        size_t end = is_casual ? i + 1 : V;

        float maxv = -10000.0f;
        for (size_t j = 0; j < end; j++) {
          maxv = std::fmax(maxv, logits[j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < end; j++) {
          probs[j] = std::exp(logits[j] - maxv);
          sum += probs[j];
        }

        for (size_t j = 0; j < end; j++) {
          probs[j] = probs[j] * (1.0f / sum);
        }

        // [end, V) is padded with 0.0f due to the causal mask
        // [V, m) is padded with 0.0f due to the padded vocab
        for (size_t j = end; j < m; j++) {
          probs[j] = 0.0f;
        }
      }
    }
  }

  static void softmax_backward(Tensor *dst, Tensor *src, bool is_casual,
                               int vocab_size) {
    src->alloc_grad();

    assert(dst->same_shape(*src));
    auto [n, m] = dst->mat();
    assert(m > 0);

    for (size_t mati = 0; mati < dst->n_mat(); mati++) {
      for (size_t i = 0; i < n; i++) {
        auto dout = reinterpret_cast<float *>(dst->grad_->data_) +
                    mati * dst->mstride() + i * m;
        auto out = reinterpret_cast<float *>(dst->data_) +
                   mati * dst->mstride() + i * m;
        auto din = reinterpret_cast<float *>(src->grad_->data_) +
                   mati * src->mstride() + i * m;
        int V = vocab_size > 0 ? vocab_size : m;
        auto end = is_casual ? i + 1 : V;

        float dsum = 0.0f;
        for (int j = 0; j < end; j++) {
          dsum += dout[j] * out[j];
        }

        for (int j = 0; j < end; j++) {
          din[j] += out[j] * (dout[j] - dsum);
        }
      }
    }
  }

  // CrossEntropy
  static void cross_entropy_forward(Tensor *dst, Tensor *src, Tensor *src1) {
    assert(dst->is_contiguous() && src->is_contiguous());
    auto losses = reinterpret_cast<float *>(dst->data_);
    auto targets = reinterpret_cast<int32_t *>(src1->data_);
    auto vs = src->vsize();

    for (size_t vi = 0; vi < src->n_vec(); vi++) {
      auto probs = reinterpret_cast<float *>(src->data_) + vi * vs;
      auto ix = targets[vi];
      losses[vi] = -std::log(probs[ix]);
    }
  }

  static void cross_entropy_backward(Tensor *dst, Tensor *src, Tensor *src1) {
    src->alloc_grad();

    auto targets = reinterpret_cast<int32_t *>(src1->data_);
    auto vs = src->vsize();

    for (size_t vi = 0; vi < src->n_vec(); vi++) {
      auto probs = reinterpret_cast<float *>(src->data_) + vi * vs;
      auto loss = (reinterpret_cast<float *>(dst->grad_->data_))[vi];
      auto din = reinterpret_cast<float *>(src->grad_->data_) + vi * vs;
      auto ix = targets[vi];
      din[ix] += -1.0f / probs[ix] * loss;
    }
  }

  static void vec_add(size_t vec_size, float *out, float *src0, float *src1) {
    for (size_t i = 0; i < vec_size; i++) {
      out[i] = src0[i] + src1[i];
    }
  }

  template <typename T>
  static void vec_fill(size_t vec_size, T *out, const T *data) {
    for (size_t i = 0; i < vec_size; i++) {
      out[i] = data[i];
    }
  }

  template <typename T> static void vec_fill(size_t vec_size, T *out, T val) {
    std::fill_n(out, vec_size, val);
  }

  static void vec_random_norm(size_t vec_size, float *out) {
    static NormalDist NORMAL_DIST;
    for (size_t i = 0; i < vec_size; i++) {
      out[i] = NORMAL_DIST();
    }
  }

  static void vec_print(size_t vec_size, TensorType type, std::byte *vec) {
    std::cout << "[";
    if (type == TensorType::F32) {
      float *vec_float = reinterpret_cast<float *>(vec);
      for (size_t i = 0; i < vec_size; i++) {
        std::cout << vec_float[i] << ((i < vec_size - 1) ? "," : "");
      }
    } else if (type == TensorType::I32) {
      int32_t *vec_int32 = reinterpret_cast<int32_t *>(vec);
      for (size_t i = 0; i < vec_size; i++) {
        std::cout << vec_int32[i] << ((i < vec_size - 1) ? "," : "");
      }
    } else {
      throw std::runtime_error("vec_print(): Not implemented");
    }
    std::cout << "]";
  }

  static std::vector<Tensor *> topo_sort(Tensor *tensor) {
    std::vector<Tensor *> sorted;
    std::unordered_map<Tensor *, int> visited;

    std::function<void(Tensor *)> dfs = [&](Tensor *t) {
      auto it = visited.find(t);
      if (it != visited.end()) {
        if (it->second == 1) {
          throw std::runtime_error("topo_sort(): Cycle detected");
        }
        return;
      }
      visited[t] = 1;
      if (t->src0_ != nullptr) {
        dfs(t->src0_);
      }
      if (t->src1_ != nullptr) {
        dfs(t->src1_);
      }
      sorted.push_back(t);
      visited[t] = 2;
    };

    dfs(tensor);
    return sorted;
  }

private:
  int n_dims_;
  int dims_[kMaxTensorDims];
  size_t strides_[kMaxTensorDims];

  TensorType type_;
  TensorOp op_;
  int op_params_[kMaxTensorOpParams];

  Tensor *grad_;
  Tensor *src0_;
  Tensor *src1_;

  std::byte *data_;

  TensorContext *ctx_;

  friend class TensorContextT<Tensor>;
};

static_assert(sizeof(Object) % kTensorMemAlign == 0,
              "Object size must be a multiple of TENSOR_MEM_ALIGN");
static_assert(sizeof(Tensor) % kTensorMemAlign == 0,
              "Tensor size must be a multiple of TENSOR_MEM_ALIGN");

} // namespace cladtorch