#pragma once

#include <cstdint>
#include <string>

#include <cuda.h>
#include "common/shape.h"

namespace marian {

/**
 * @brief Represents the size of each dimension in a tensor.
 *
 * Note: this class currently is hard-coded to four dimensions.
 */

struct ShapeGPU {
  int shape_[SHAPE_SIZE];
  int stride_[SHAPE_SIZE];
  int bstride_[SHAPE_SIZE];

  /**
   * @brief Constructs a default shape.
   *
   * This default shape has four dimensions.
   * The size of each dimension is 1.
   */
  ShapeGPU() : shape_{1, 1, 1, 1}, stride_{1, 1, 1, 1}, bstride_{0, 0, 0, 0} {}

  /**
   * @brief Constructs a shape.
   *
   * @param i A list of integers representing the size of each dimension.
   */
  ShapeGPU(std::initializer_list<int> il) : ShapeGPU() {
    std::copy(il.begin(), il.end(), begin());
    updateStrides();
  }

  ShapeGPU(const ShapeGPU& shape) : ShapeGPU() {
    shape_[0] = shape.shape_[0];
    shape_[1] = shape.shape_[1];
    shape_[2] = shape.shape_[2];
    shape_[3] = shape.shape_[3];
    updateStrides();
  }

  ShapeGPU(const Shape& shape) : ShapeGPU() {
    shape_[0] = shape[0];
    shape_[1] = shape[1];
    shape_[2] = shape[2];
    shape_[3] = shape[3];
    updateStrides();
  }

  void updateStrides() {
    stride_[0] = shape_[1];
    stride_[1] = 1;
    stride_[2] = shape_[0] * shape_[1];
    stride_[3] = shape_[0] * shape_[1] * shape_[2];

    bstride_[0] = shape_[0] == 1 ? 0 : stride_[0];
    bstride_[1] = shape_[1] == 1 ? 0 : stride_[1];
    bstride_[2] = shape_[2] == 1 ? 0 : stride_[2];
    bstride_[3] = shape_[3] == 1 ? 0 : stride_[3];
  }

  inline void set(int i, int dim) {
    shape_[i] = dim;
    updateStrides();
  }
  /**
   * @brief Gets a reference to the int representing the size of the
   * <code>i</code>th dimension represented by this object.
   *
   * @return a reference to the int representing the size of the
   * <code>i</code>th dimension represented by this object
   */
  __host__ __device__ inline int dim(int i) { return shape_[i]; }

  /**
   * @brief Gets the size of the <code>i</code>th dimension represented by this
   * object.
   *
   * @return the size of the <code>i</code>th dimension represented by this
   * object
   */
  __host__ __device__ inline int dim(int i) const {
    return const_cast<ShapeGPU&>(*this).dim(i);
  }

  /**
   * @brief Gets a reference to the int representing the size of the
   * <code>i</code>th dimension represented by this object.
   *
   * @return a reference to the int representing the size of the
   * <code>i</code>th dimension represented by this object
   */
  __host__ __device__ inline int operator[](int i) { return dim(i); }

  /**
   * @brief Gets the size of the <code>i</code>th dimension represented by this
   * object.
   *
   * @return the size of the <code>i</code>th dimension represented by this
   * object
   */
  __host__ __device__ inline int operator[](int i) const { return dim(i); }

  __host__ __device__ inline int stride(int i) const { return stride_[i]; }

  __host__ __device__ inline int bstride(int i) const { return bstride_[i]; }

  /**
   * @brief Gets the number of dimensions represented by this object
   *
   * @return the number of dimensions represented by this object
   */
  __host__ __device__ inline size_t size() const { return SHAPE_SIZE; }

  /**
   * @brief Gets the total number of elements in a tensor of this shape.
   *
   * For example, if this shape represents a 5x100 tensor, this method would
   * return 500.
   *
   * @return the total number of elements in a tensor of this shape
   */
  __host__ __device__ inline int elements() const {
    return shape_[0] * shape_[1] * shape_[2] * shape_[3];
  }

  __host__ __device__ inline int index(int* d) const {
    return d[0] * stride(0) + d[1] * stride(1) + d[2] * stride(2)
           + d[3] * stride(3);
  }

  __host__ __device__ inline int bindex(int* d) const {
    return d[0] * bstride(0) + d[1] * bstride(1) + d[2] * bstride(2)
           + d[3] * bstride(3);
  }

  __host__ __device__ inline void dims(int i, int* d) const {
    d[0] = (i / stride_[0]) % shape_[0];
    d[1] = (i / stride_[1]) % shape_[1];
    d[2] = (i / stride_[2]) % shape_[2];
    d[3] = i / stride_[3];
  }

  /** @brief Gets a pointer to an int that specifies the size of the first
   * dimension represented by this object */
  int* begin() { return shape_; }

  /** @brief Gets a pointer to an int that specifies the size of the last
   * dimension represented by this object */
  int* end() { return shape_ + SHAPE_SIZE; }

  /** @brief Gets a const pointer to an int that specifies the size of the first
   * dimension represented by this object */
  const int* begin() const { return shape_; }

  /** @brief Gets a const pointer to an int that specifies the size of the last
   * dimension represented by this object */
  const int* end() const { return shape_ + SHAPE_SIZE; }

  __host__ __device__ bool operator==(const ShapeGPU& other) const {
    return
      shape_[0] == other[0] &&
      shape_[1] == other[1] &&
      shape_[2] == other[2] &&
      shape_[3] == other[3];
  }

  __host__ __device__ bool operator!=(const ShapeGPU& other) const { return !(*this == other); }
};
}
