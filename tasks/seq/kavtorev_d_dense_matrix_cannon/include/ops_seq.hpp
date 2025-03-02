#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_dense_matrix_cannon_seq {

class MatrixMultiplySequential : public ppc::core::Task {
 public:
  explicit MatrixMultiplySequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  int n_ = 0;
  int block_size_ = 0;

  void CannonMatrixMultiply();
};

}  // namespace kavtorev_d_dense_matrix_cannon_seq