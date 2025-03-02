#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_dense_matrix_cannon/include/ops_seq.hpp"

using namespace kavtorev_d_dense_matrix_cannon_seq;

TEST(kavtorev_d_dense_matrix_cannon_seq, SimpleData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 4;
  int block_size = 2;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> B = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> C(n * n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C = {80.0,  70.0,  60.0,  50.0,  240.0, 214.0, 188.0, 162.0,
                                    400.0, 358.0, 316.0, 274.0, 560.0, 502.0, 444.0, 386.0};

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_seq, ValidationFailureTestSize) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 4;
  int block_size = 2;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {1.0, 2.0, 3.0};
  std::vector<double> C(n * n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(4);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(3);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(kavtorev_d_dense_matrix_cannon_seq, RandomDataSmall) {
  int n = 4;
  int block_size = 2;
  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }

  std::vector<double> C(n * n, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        expected_C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_seq, RandomDataLarge) {
  int n = 8;
  int block_size = 2;
  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(gen);
    B[i] = dist(gen);
  }

  std::vector<double> C(n * n, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        expected_C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_seq, AlreadySortedData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 4;
  int block_size = 2;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> B = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> C(n * n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C = {80.0,  70.0,  60.0,  50.0,  240.0, 214.0, 188.0, 162.0,
                                    400.0, 358.0, 316.0, 274.0, 560.0, 502.0, 444.0, 386.0};

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_seq, ReverseSortedData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 4;
  int block_size = 2;
  std::vector<double> A = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> B = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> C(n * n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C = {80.0,  90.0,  100.0, 110.0, 240.0, 278.0, 316.0, 354.0,
                                    400.0, 466.0, 532.0, 598.0, 560.0, 654.0, 748.0, 842.0};

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}

TEST(kavtorev_d_dense_matrix_cannon_seq, AllOnesData) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  int n = 4;
  int block_size = 2;
  std::vector<double> A(n * n, 1.0);
  std::vector<double> B(n * n, 1.0);
  std::vector<double> C(n * n, 0.0);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  task_data_seq->inputs_count.emplace_back(1);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(n * n);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(n * n);

  MatrixMultiplySequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::vector<double> expected_C(n * n, static_cast<double>(n));

  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(expected_C[i], C[i], 1e-12);
  }
}