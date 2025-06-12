import tensorrt as trt
import numpy as np

def build_flexible_sam2_engine():
    """使用 Python API 构建灵活的 SAM2 引擎"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # 解析 ONNX
        print("Loading ONNX model...")
        with open("sam2.1_large.onnx", 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"Error {error}: {parser.get_error(error)}")
                return False
        
        # 创建 builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB
        
        # 打印网络信息
        print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}")
        
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"Output {i}: {output_tensor.name}, shape: {output_tensor.shape}")
        
        # 创建优化配置文件
        profile = builder.create_optimization_profile()
        
        # 设置动态输入的形状范围
        profile.set_shape("image_embeddings", (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
        profile.set_shape("high_res_features1", (1, 32, 256, 256), (1, 32, 256, 256), (1, 32, 256, 256))
        profile.set_shape("high_res_features2", (1, 64, 128, 128), (1, 64, 128, 128), (1, 64, 128, 128))
        profile.set_shape("point_coords", (1, 1, 2), (1, 5, 2), (1, 10, 2))
        profile.set_shape("point_labels", (1, 1), (1, 5), (1, 10))
        profile.set_shape("mask_input", (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
        profile.set_shape("has_mask_input", (1,), (1,), (1,))
        
        # 关键修复：为形状张量 orig_im_size 设置具体的值
        # 使用你想要的目标尺寸 1024x1797
        profile.set_shape("orig_im_size", (2,), (2,), (2,))
        
        # 为形状张量设置具体的值
        target_size = np.array([1920, 1080], dtype=np.int64)
        profile.set_shape_input("orig_im_size", target_size, target_size, target_size)
        
        # 添加配置文件
        config.add_optimization_profile(profile)
        
        print("Building TensorRT engine... This may take several minutes.")
        print(f"Target output size: {target_size}")
        
        # 构建引擎
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("Failed to build engine")
            return False
        
        # 保存引擎
        engine_path = "sam2.1_large_1024x1797_python.engine"
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ Engine successfully built and saved to: {engine_path}")
        
        # 验证引擎
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine:
            print("✅ Engine validation successful!")
            print(f"Number of IO tensors: {engine.num_io_tensors}")
            
            # 打印张量信息
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                tensor_shape = engine.get_tensor_shape(tensor_name)
                tensor_mode = engine.get_tensor_mode(tensor_name)
                io_type = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
                print(f"  {io_type}: {tensor_name} -> {tensor_shape}")
            
            return True
        else:
            print("❌ Engine validation failed!")
            return False

if __name__ == "__main__":
    success = build_flexible_sam2_engine()
    if not success:
        print("Engine build failed!")