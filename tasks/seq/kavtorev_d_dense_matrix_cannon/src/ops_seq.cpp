#include "seq/kavtorev_d_dense_matrix_cannon/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace kavtorev_d_dense_matrix_cannon_seq;

bool MatrixMultiplySequential::PreProcessingImpl() {
  n_ = *(reinterpret_cast<int*>(task_data->inputs[0]));
  block_size_ = *(reinterpret_cast<int*>(task_data->inputs[1]));

  A_.resize(n_ * n_);
  B_.resize(n_ * n_);
  C_.resize(n_ * n_, 0.0);

  auto* arrA = reinterpret_cast<double*>(task_data->inputs[2]);
  auto* arrB = reinterpret_cast<double*>(task_data->inputs[3]);
  std::copy(arrA, arrA + n_ * n_, A_.begin());
  std::copy(arrB, arrB + n_ * n_, B_.begin());

  return true;
}

bool MatrixMultiplySequential::ValidationImpl() {
  bool is_valid = true;
  if (task_data->inputs_count[2] != task_data->inputs_count[3]) {
    is_valid = false;
  }

  return is_valid;
}

bool MatrixMultiplySequential::RunImpl() {
  CannonMatrixMultiply();
  return true;
}

bool MatrixMultiplySequential::PostProcessingImpl() {
  auto* out = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(C_.begin(), C_.end(), out);
  return true;
}

void MatrixMultiplySequential::CannonMatrixMultiply() {
  int sqrt_size = n_ / block_size_;
  std::vector<double> tempA(block_size_ * block_size_);
  std::vector<double> tempB(block_size_ * block_size_);

  for (int i = 0; i < sqrt_size; ++i) {
    for (int j = 0; j < sqrt_size; ++j) {
      int A_src = (i + j) % sqrt_size;
      int B_src = (i + j) % sqrt_size;

      for (int k = 0; k < block_size_; ++k) {
        for (int l = 0; l < block_size_; ++l) {
          tempA[k * block_size_ + l] = A_[(i * block_size_ + k) * n_ + (A_src * block_size_ + l)];
          tempB[k * block_size_ + l] = B_[(B_src * block_size_ + k) * n_ + (j * block_size_ + l)];
        }
      }

      for (int step = 0; step < sqrt_size; ++step) {
        for (int k = 0; k < block_size_; ++k) {
          for (int l = 0; l < block_size_; ++l) {
            for (int m = 0; m < block_size_; ++m) {
              C_[(i * block_size_ + k) * n_ + (j * block_size_ + l)] +=
                  tempA[k * block_size_ + m] * tempB[m * block_size_ + l];
            }
          }
        }

        int A_dest = (A_src + 1) % sqrt_size;
        int B_dest = (B_src + 1) % sqrt_size;

        for (int k = 0; k < block_size_; ++k) {
          for (int l = 0; l < block_size_; ++l) {
            tempA[k * block_size_ + l] = A_[(i * block_size_ + k) * n_ + (A_dest * block_size_ + l)];
            tempB[k * block_size_ + l] = B_[(B_dest * block_size_ + k) * n_ + (j * block_size_ + l)];
          }
        }

        A_src = A_dest;
        B_src = B_dest;
      }
    }
  }
}