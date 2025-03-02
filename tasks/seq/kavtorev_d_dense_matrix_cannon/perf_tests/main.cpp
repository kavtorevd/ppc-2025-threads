#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kavtorev_d_dense_matrix_cannon/include/ops_seq.hpp"

using namespace kavtorev_d_dense_matrix_cannon_seq;

TEST(kavtorev_d_dense_matrix_cannon_seq, test_pipeline_run) {
  int n = 512;          // Размер матрицы (n x n)
  int block_size = 64;  // Размер блока
  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  std::vector<double> C(n * n, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < n * n; ++i) {
      A[i] = dist(gen);
      B[i] = dist(gen);
    }
  }

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

  auto test_task_sequential = std::make_shared<MatrixMultiplySequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Проверка корректности результата
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

TEST(kavtorev_d_dense_matrix_cannon_seq, test_task_run) {
  int n = 512;          // Размер матрицы (n x n)
  int block_size = 64;  // Размер блока
  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  std::vector<double> C(n * n, 0.0);

  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1e9, 1e9);
    for (int i = 0; i < n * n; ++i) {
      A[i] = dist(gen);
      B[i] = dist(gen);
    }
  }

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

  auto test_task_sequential = std::make_shared<MatrixMultiplySequential>(task_data_seq);
  ASSERT_TRUE(test_task_sequential->ValidationImpl());
  test_task_sequential->PreProcessingImpl();
  test_task_sequential->RunImpl();
  test_task_sequential->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Проверка корректности результата
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