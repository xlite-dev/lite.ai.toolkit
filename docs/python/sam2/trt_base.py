import numpy as np
import tensorrt as trt
import time
from abc import ABC, abstractmethod
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    """主机和设备内存管理辅助类"""
    
    def __init__(self, host_mem, device_mem, name=""):
        self.host = host_mem
        self.device = device_mem
        self.name = name

    def __str__(self):
        return f"Name: {self.name}\nHost:\n{self.host}\nDevice Ptr:\n{self.device}"

    def __repr__(self):
        return self.__str__()


class BaseTRT(ABC):
    """TensorRT推理引擎基类
    
    负责TensorRT引擎的初始化、内存分配和推理执行。
    子类只需实现preprocess和postprocess方法即可。
    """
    
    def __init__(self, plan_path):
        """初始化TensorRT推理引擎
        
        Args:
            plan_path (str): TensorRT引擎计划文件路径
        """
        self.plan_path = plan_path
        
        # 初始化TensorRT相关组件
        self._init_tensorrt()
        
    def _init_tensorrt(self):
        """初始化TensorRT引擎和执行上下文"""
        # 创建Logger和Runtime
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, "")
        self.runtime = trt.Runtime(self.trt_logger)

        # 加载引擎
        try:
            with open(self.plan_path, "rb") as f:
                engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            print(f"ERROR: Failed to deserialize engine: {e}")
            raise SystemExit(f"Failed to load TensorRT engine from {self.plan_path}")

        print(f"Engine {self.plan_path} loaded successfully.")
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 初始化内存管理相关变量
        self.bindings_addrs = []  # 存储execute_v2使用的整数地址
        self.inputs_hdm = {}      # 按名称存储输入的HostDeviceMem
        self.outputs_hdm = {}     # 按名称存储输出的HostDeviceMem
        self.output_shapes_map = {} # 按名称存储输出形状

        # 处理输入输出张量
        self._process_io_tensors()
        
    def _process_io_tensors(self):
        """处理引擎的输入和输出张量，分配相应的内存"""
        try:
            num_io_tensors = self.engine.num_io_tensors
            print(f"Engine has {num_io_tensors} IO tensors")
            
            # 获取所有张量名称
            tensor_names = []
            for i in range(num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_names.append(tensor_name)
            
            print(f"Tensor names: {tensor_names}")
            
            # 处理每个张量
            for tensor_name in tensor_names:
                # 判断是否为输入张量（基于命名约定）
                is_input = self._is_input_tensor(tensor_name,tensor_names)
                
                shape = self.engine.get_tensor_shape(tensor_name)
                dtype = self.engine.get_tensor_dtype(tensor_name)
                
                
                print(f"Tensor: Name='{tensor_name}', IsInput={is_input}, Shape={shape}, DType={dtype}")
                
                
                # 将动态维度 -1 替换为 4
                if -1 in shape:
                    shape = tuple(4 if dim == -1 else dim for dim in shape)
                    print(f"Modified dynamic shape for {tensor_name}: original had -1, replaced with 4")

                print(f"Tensor: Name='{tensor_name}', IsInput={is_input}, Shape={shape}, DType={dtype}")
                
                
                # 分配主机和设备内存
                host_mem = np.empty(abs(trt.volume(shape)), dtype=trt.nptype(dtype))
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings_addrs.append(int(device_mem))
                
                hdm = HostDeviceMem(host_mem, device_mem, tensor_name)
                
                # 根据类型存储到相应的字典中
                if is_input:
                    self.inputs_hdm[tensor_name] = hdm
                    self._setup_input_properties(tensor_name, shape)
                else:
                    self.outputs_hdm[tensor_name] = hdm
                    self.output_shapes_map[tensor_name] = shape
                    self._setup_output_properties(tensor_name)
                    
            print(f"Processed {len(self.inputs_hdm)} inputs and {len(self.outputs_hdm)} outputs")
            
        except Exception as e:
            print(f"CRITICAL ERROR processing TensorRT tensors: {e}")
            import traceback
            traceback.print_exc()
            raise SystemExit(f"Failed to initialize TensorRT engine: {e}")
        
        # 验证至少存在一个输入和输出
        self._validate_io_tensors()
        
    def _is_input_tensor(self, tensor_name,tensor_names):
        """判断张量是否为输入张量
        
        Args:
            tensor_name (str): 张量名称
            
        Returns:
            bool: 如果是输入张量返回True，否则返回False
        """
        if len(tensor_names) == 4:
            if tensor_name.lower() == "input":
                return True
            else:
                return False
        
        if len(tensor_names) > 4:
            if tensor_name.lower() == "image_embeddings" or tensor_name.lower() == "high_res_features1" or tensor_name.lower() == "high_res_features2" or tensor_name.lower() == "point_coords" or tensor_name.lower() == "point_labels" or \
                    tensor_name.lower() == "mask_input" or tensor_name.lower() == "has_mask_input" or tensor_name.lower() == "orig_im_size":
                return True
            else:
                return False
    
    def _setup_input_properties(self, tensor_name, shape):
        """设置输入张量的相关属性
        
        Args:
            tensor_name (str): 张量名称
            shape (tuple): 张量形状
        """
        # 设置主要输入张量
        if not hasattr(self, 'input_name_from_engine'):
            self.input_name_from_engine = tensor_name
            self.input_shape_from_engine = shape
            
            # 假设NCHW格式，提取高度和宽度
            if len(shape) >= 4:
                self.input_height = int(shape[2])
                self.input_width = int(shape[3])
            else:
                print(f"WARNING: Input shape {shape} doesn't match expected NCHW format")
                # 设置默认值，可由子类覆盖
                self.input_height = 640
                self.input_width = 640
    
    def _setup_output_properties(self, tensor_name):
        """设置输出张量的相关属性
        
        Args:
            tensor_name (str): 张量名称
        """
        # 设置主要输出张量
        if not hasattr(self, 'output_name_from_engine'):
            self.output_name_from_engine = tensor_name
    
    def _validate_io_tensors(self):
        """验证输入输出张量的有效性"""
        if not self.inputs_hdm:
            raise SystemExit("ERROR: No input tensors found in engine")
            
        if not self.outputs_hdm:
            raise SystemExit("ERROR: No output tensors found in engine")
        
        # 如果没有找到主要输入，使用第一个输入
        if not hasattr(self, 'input_name_from_engine'):
            self.input_name_from_engine = list(self.inputs_hdm.keys())[0]
            self.input_shape_from_engine = self.engine.get_tensor_shape(self.input_name_from_engine)
            print(f"WARNING: Using '{self.input_name_from_engine}' as primary input")
            
            if len(self.input_shape_from_engine) >= 4:
                self.input_height = int(self.input_shape_from_engine[2])
                self.input_width = int(self.input_shape_from_engine[3])
            else:
                self.input_height = 640
                self.input_width = 640
                
        # 如果没有找到主要输出，使用第一个输出
        if not hasattr(self, 'output_name_from_engine'):
            self.output_name_from_engine = list(self.outputs_hdm.keys())[0]
            print(f"WARNING: Using '{self.output_name_from_engine}' as primary output")
            
        print(f"Configured - Input: '{self.input_name_from_engine}', Output: '{self.output_name_from_engine}'")
        print(f"Input dimensions: {self.input_height}x{self.input_width}")

    def infer(self, input_data):
        """执行TensorRT推理
        
        Args:
            input_data (np.ndarray): 预处理后的输入数据
            
        Returns:
            np.ndarray: 推理结果
        """
        # 将输入数据复制到主机内存，然后传输到设备
        active_input_hdm = self.inputs_hdm[self.input_name_from_engine]
        np.copyto(active_input_hdm.host, input_data.ravel())
        cuda.memcpy_htod(active_input_hdm.device, active_input_hdm.host)

        # 执行推理
        if not self.context.execute_v2(bindings=self.bindings_addrs):
            raise RuntimeError("TensorRT execute_v2() failed")

        # 将结果从设备复制回主机
        active_output_hdm = self.outputs_hdm[self.output_name_from_engine]
        cuda.memcpy_dtoh(active_output_hdm.host, active_output_hdm.device)
        
        # 重塑输出为正确的形状
        return active_output_hdm.host.reshape(self.output_shapes_map[self.output_name_from_engine])

    def detect(self, image):
        """执行完整的检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果
        """
        # 预处理
        preprocessed_input = self.preprocess(image)
        
        # 推理
        raw_output = self.infer(preprocessed_input)
        
        # 后处理
        return self.postprocess(raw_output, image)
    
    @abstractmethod
    def preprocess(self, image):
        """图像预处理抽象方法，子类必须实现
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像张量
        """
        pass
    
    @abstractmethod
    def postprocess(self, output, original_image):
        """模型输出后处理抽象方法，子类必须实现
        
        Args:
            output (np.ndarray): 模型原始输出
            original_image: 原始输入图像
            
        Returns:
            处理后的结果
        """
        pass
    
    def benchmark(self, image, num_runs=10):
        """对模型进行性能基准测试
        
        Args:
            image: 输入图像
            num_runs (int): 运行次数
            
        Returns:
            tuple: (平均推理时间, 最后一次的推理结果)
        """
        # 预热运行
        print("Performing a warm-up inference run...")
        try:
            _ = self.detect(image.copy())
            print("Warm-up run complete.")
        except Exception as e:
            print(f"Warm-up run failed: {e}")
            return None, None
        
        total_time = 0
        last_result = None
        
        print(f"Performing {num_runs} timed inference runs...")
        for i in range(num_runs):
            start_time = time.time()
            result = self.detect(image.copy())
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            
            if i == num_runs - 1:
                last_result = result
                
            print(f"Run {i+1}/{num_runs} - Inference time: {inference_time*1000:.2f} ms")
            
        avg_time = total_time / num_runs
        print(f"\n--- Benchmark Summary ---")
        print(f"Average inference time over {num_runs} runs: {avg_time*1000:.2f} ms")
        
        return avg_time, last_result
    
    def get_input_shape(self):
        """获取输入张量形状
        
        Returns:
            tuple: 输入张量形状
        """
        return self.input_shape_from_engine
    
    def get_output_shapes(self):
        """获取所有输出张量形状
        
        Returns:
            dict: 输出张量名称到形状的映射
        """
        return self.output_shapes_map.copy()
    
    def get_tensor_names(self):
        """获取所有张量名称
        
        Returns:
            dict: 包含输入和输出张量名称的字典
        """
        return {
            'inputs': list(self.inputs_hdm.keys()),
            'outputs': list(self.outputs_hdm.keys())
        }

    def __del__(self):
        """析构函数，清理资源"""
        try:
            # 检查 CUDA 上下文是否还有效
            try:
                import pycuda.driver as cuda
                current_context = cuda.Context.get_current()
                cuda_valid = current_context is not None
            except:
                cuda_valid = False
            
            # 只有在 CUDA 上下文有效时才清理设备内存
            if cuda_valid:
                # 清理输入设备内存
                if hasattr(self, 'inputs_hdm') and self.inputs_hdm:
                    for hdm in self.inputs_hdm.values():
                        try:
                            if hasattr(hdm, 'device') and hdm.device:
                                hdm.device.free()
                        except:
                            pass
                
                # 清理输出设备内存
                if hasattr(self, 'outputs_hdm') and self.outputs_hdm:
                    for hdm in self.outputs_hdm.values():
                        try:
                            if hasattr(hdm, 'device') and hdm.device:
                                hdm.device.free()
                        except:
                            pass
            
            # 清理 TensorRT 对象（即使 CUDA 上下文无效也要清理）
            if hasattr(self, 'context'):
                try:
                    del self.context
                except:
                    pass
            
            if hasattr(self, 'engine'):
                try:
                    del self.engine
                except:
                    pass
                    
            if hasattr(self, 'runtime'):
                try:
                    del self.runtime
                except:
                    pass
                    
        except:
            pass