#include "cpp_api_test.h"
#include "nvToolsExt.h"

TEST_P(CppAPITests, CompiledModuleIsClose) {

  // cudaSetDevice(2); // A30
  cudaSetDevice(0); // A100

  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  std::vector<torch_tensorrt::Input> shapes;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    auto in = at::randint(5, input_shapes[i], {at::kCUDA}).to(input_types[i]);
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
    auto in_spec = torch_tensorrt::Input(input_shapes[i]);
    in_spec.dtype = input_types[i];
    shapes.push_back(in_spec);
    std::cout << in_spec << std::endl;
  }
  mod.to(at::kHalf);
  // mod.to(at::kFloat);
  torch::jit::IValue jit_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(mod, jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  if (jit_results_ivalues.isTuple()) {
    auto tuple = jit_results_ivalues.toTuple();
    for (auto t : tuple->elements()) {
      jit_results.push_back(t.toTensor());
    }
  } else {
    jit_results.push_back(jit_results_ivalues.toTensor());
  }

  auto spec = torch_tensorrt::ts::CompileSpec(shapes);
  spec.truncate_long_and_double = true;
  spec.enabled_precisions = {torch_tensorrt::DataType::kHalf};
  // spec.enabled_precisions = {torch_tensorrt::DataType::kFloat};
  
  auto trt_mod = torch_tensorrt::ts::compile(mod, spec);
  torch::jit::IValue trt_results_ivalues;

  trt_mod.to(at::kCUDA);
  // cudaSetDevice(3); // T4
  // cudaEvent_t start_event;
  // cudaEvent_t stop_event;
  // cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
  // cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
  cudaDeviceSynchronize();
  auto start_time = std::chrono::steady_clock::now();
  nvtxRangePushA("Inference");
  for (int i = 0; i < 100; i++) {
    // trt_results_ivalues = torch_tensorrt::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
    trt_results_ivalues = trt_mod.forward(trt_inputs_ivalues);
  }
  nvtxRangePop();
  cudaDeviceSynchronize();
  auto stop_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_s = stop_time - start_time;
  std::cout <<  time_s.count() * 1e3  / 100.0 << "ms" << std::endl;; // ms

  // std::vector<at::Tensor> trt_results;
  // if (trt_results_ivalues.isTuple()) {
  //   auto tuple = trt_results_ivalues.toTuple();
  //   for (auto t : tuple->elements()) {
  //     trt_results.push_back(t.toTensor());
  //   }
  // } else {
  //   trt_results.push_back(trt_results_ivalues.toTensor());
  // }

  // for (size_t i = 0; i < trt_results.size(); i++) {
  //   ASSERT_TRUE(
  //       torch_tensorrt::tests::util::cosineSimEqual(jit_results[i], trt_results[i].reshape_as(jit_results[i]), 0.99));
  // }
}

#ifndef DISABLE_TEST_IN_CI

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    CppAPITests,
    testing::Values(
        // PathAndInput({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5}),
        PathAndInput({"tests/modules/resnet50_traced.jit.pt", {{1, 3, 224, 224}}, {at::kHalf}, 2e-5})
        // PathAndInput({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5})
        // PathAndInput({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 2e-5}),
        // PathAndInput({"tests/modules/efficientnet_b0_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 8e-3}),
        // PathAndInput({"tests/modules/bert_base_uncased_traced.jit.pt", {{1, 14}, {1, 14}}, {at::kInt, at::kInt}, 8e-2}),
        // PathAndInput({"tests/modules/vit_scripted.jit.pt", {{1, 3, 224, 224}}, {at::kFloat}, 8e-2})
        ));

#endif
