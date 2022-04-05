#include <algorithm>

#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "torch_tensorrt/torch_tensorrt.h"

namespace torch_tensorrt {
// Defined in types.cpp
nvinfer1::DataType toTRTDataType(DataType value);
nvinfer1::TensorFormat toTRTTensorFormat(TensorFormat value);
torchtrt::core::ir::Input to_internal_input(Input& i);
std::vector<torchtrt::core::ir::Input> to_vec_internal_inputs(std::vector<Input>& external);
torchtrt::core::runtime::CudaDevice to_internal_cuda_device(Device device);

namespace torchscript {
CompileSpec::CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
  graph_inputs.flattened_inputs = inputs;
}

CompileSpec::CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
  graph_inputs.flattened_inputs = inputs;
}

void flatten_dfs(std::vector<torchtrt::core::ir::Input>& flattened_inputs, torch::jit::IValue input_ivalue, torch::jit::IValue& converted_ivalue) {
    if (input_ivalue.isTuple()) {
      auto input_tuple = input_ivalue.toTuple();
      std::vector<torch::jit::IValue> converted_elements;
      for (auto item: input_tuple->elements()) {
        torch::jit::IValue converted_item;
        flatten_dfs(flattened_inputs, item, converted_item);
        converted_elements.push_back(converted_item);
        auto tuple_ptr = c10::ivalue::Tuple::create(converted_elements);
        converted_ivalue = torch::jit::IValue(tuple_ptr);
      }
    } else if(input_ivalue.isList()) {
      auto input_list = input_ivalue.toList().vec();
      c10::TypePtr type = input_list[0].type();
      auto converted_elements = c10::impl::GenericList(type);
      // std::vector<torch::jit::IValue> converted_elements;
      for (auto item: input_list) {
        torch::jit::IValue converted_item;
        flatten_dfs(flattened_inputs, item, converted_item);
        converted_elements.push_back(converted_item);
      }
      converted_ivalue = torch::jit::IValue(converted_elements);
    } else if(input_ivalue.isCustomClass()) {
      torchtrt::core::ir::Input cur_input = to_internal_input(*(input_ivalue.toCustomClass<torchtrt::Input>()));
      flattened_inputs.push_back(cur_input);
      converted_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::core::ir::Input>(cur_input)));
    }
}

torch_tensorrt::core::ir::GraphInputs to_internal_graph_inputs(GraphInputs external_graph_input) {
  torch_tensorrt::core::ir::GraphInputs internal_graph_input;

  // flattened version
  if (external_graph_input.flattened_inputs.size() > 0) {
    // std::vector<torch::jit::IValue> input_shape_list;
    auto empty_ivalue = torch::jit::IValue(c10::make_intrusive<torchtrt::core::ir::Input>(torchtrt::core::ir::Input()));
    c10::TypePtr type = empty_ivalue.type();
    auto input_shape_list = c10::impl::GenericList(type);
    std::vector<torchtrt::core::ir::Input> internal_input = to_vec_internal_inputs(external_graph_input.flattened_inputs);
    for (auto input_shape: internal_input) {
      auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torchtrt::core::ir::Input>(input_shape)));
      input_shape_list.push_back(input_shape_ivalue);
    }

    torch::jit::IValue input_signature(input_shape_list);
    internal_graph_input.flattened_inputs = internal_input;
    internal_graph_input.input_signature = input_signature;
    
  }
  // nested version
  else {
    std::vector<torchtrt::core::ir::Input> flattened_inputs;
    torch::jit::IValue input_signature;
    flatten_dfs(flattened_inputs, external_graph_input.input_signature, input_signature);
    internal_graph_input.flattened_inputs = flattened_inputs;
    internal_graph_input.input_signature = input_signature;
    printf("in nested version branch\n");

  }
  return internal_graph_input;
}

torchtrt::core::CompileSpec to_internal_compile_spec(CompileSpec external) {
  torchtrt::core::CompileSpec internal(to_vec_internal_inputs(external.inputs));
  internal.graph_inputs = to_internal_graph_inputs(external.graph_inputs);
  internal.inputs = internal.graph_inputs.flattened_inputs;

  for (auto p : external.enabled_precisions) {
    internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
  }

  internal.convert_info.engine_settings.sparse_weights = external.sparse_weights;
  internal.convert_info.engine_settings.disable_tf32 = external.disable_tf32;
  internal.convert_info.engine_settings.refit = external.refit;
  internal.convert_info.engine_settings.debug = external.debug;
  internal.convert_info.engine_settings.truncate_long_and_double = external.truncate_long_and_double;
  internal.convert_info.engine_settings.device.allow_gpu_fallback = external.device.allow_gpu_fallback;

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_ops.size() > 0)),
      "require_full_compilation is enabled however the list of ops to run in torch is not empty (Found "
          << external.torch_executed_ops.size() << " ops)");

  TORCHTRT_CHECK(
      !(external.require_full_compilation && (external.torch_executed_modules.size() > 0)),
      "require_full_compilation is enabled however the list of modules to run in torch is not empty (Found "
          << external.torch_executed_modules.size() << " modules)");

  internal.partition_info.enabled = !external.require_full_compilation;
  internal.partition_info.min_block_size = external.min_block_size;
  internal.partition_info.forced_fallback_operators = std::move(external.torch_executed_ops);
  internal.partition_info.truncate_long_and_double = external.truncate_long_and_double;
  internal.lower_info.forced_fallback_modules = std::move(external.torch_executed_modules);

  switch (external.device.device_type) {
    case Device::DeviceType::kDLA:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kDLA;
      break;
    case Device::DeviceType::kGPU:
    default:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kGPU;
  }

  switch (external.capability) {
    case EngineCapability::kSAFETY:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_SAFETY;
      break;
    case EngineCapability::kDLA_STANDALONE:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_DLA_STANDALONE;
      break;
    case EngineCapability::kSTANDARD:
    default:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_STANDARD;
  }

  internal.convert_info.engine_settings.device.gpu_id = external.device.gpu_id;
  internal.convert_info.engine_settings.device.dla_core = external.device.dla_core;
  internal.convert_info.engine_settings.num_min_timing_iters = external.num_min_timing_iters;
  internal.convert_info.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;
  internal.convert_info.engine_settings.workspace_size = external.workspace_size;

  if (internal.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) !=
      internal.convert_info.engine_settings.enabled_precisions.end()) {
    if (external.ptq_calibrator) {
      internal.convert_info.engine_settings.calibrator = external.ptq_calibrator;
    } else {
      internal.lower_info.unfreeze_module = true;
      internal.lower_info.disable_cse = true;
      internal.convert_info.engine_settings.calibrator = nullptr;
    }
  } else {
    internal.convert_info.engine_settings.calibrator = nullptr;
  }

  return internal;
}
} // namespace torchscript
} // namespace torch_tensorrt
