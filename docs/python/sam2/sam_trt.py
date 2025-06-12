import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
from trt_base import BaseTRT


class SAM2Encoder(BaseTRT):
    """SAM2 编码器 TensorRT 实现"""
    
    def __init__(self, plan_path: str):
        super().__init__(plan_path)
        self.mode = "SAM2"  # 或 "SAM"
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """SAM2 编码器预处理"""
        if self.mode == "SAM2":
            return self._prepare_float_input(image)
        else:
            return self._prepare_uint8_input(image)
    
    def postprocess(self, output, original_image):
        """编码器的后处理比较简单，直接返回特征"""
        # output 是一个包含所有输出的数组
        # 对于 SAM2，通常有 3 个输出：image_embed, high_res_feats_0, high_res_feats_1
        return output
    
    def _prepare_float_input(self, image: np.ndarray) -> np.ndarray:
        """为 SAM2 准备 float 输入张量"""
        # ImageNet normalization for SAM2
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        
        # Apply normalization
        image_normalized = (image_float - mean) / std
        image_normalized = image_normalized.astype(np.float32)
        
        # Convert to NCHW format
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor.astype(np.float32)
    
    def _prepare_uint8_input(self, image: np.ndarray) -> np.ndarray:
        """为 SAM1 准备 uint8 输入张量"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to NCHW format
        image_tensor = np.transpose(image_rgb, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor.astype(np.uint8)


class SAM2Decoder(BaseTRT):
    """SAM2 解码器 TensorRT 实现"""
    
    def __init__(self, plan_path: str):
        super().__init__(plan_path)
        self.mode = "SAM2"
        self.previous_masks = []
    
    def preprocess(self, decoder_inputs: dict) -> np.ndarray:
        """解码器预处理 - 直接使用传入的输入字典"""
        # 这里需要按照 TensorRT 引擎期望的顺序返回输入
        # 通常解码器有多个输入，我们需要按顺序组织它们
        
        # 获取输入张量的名称顺序
        input_names = list(self.inputs_hdm.keys())
        
        # 按照引擎期望的顺序准备输入
        # 对于主要输入（通常是第一个），我们返回它
        # 其他输入会在 infer_decoder 方法中处理
        main_input_name = self.input_name_from_engine
        return decoder_inputs[main_input_name]
    
    def postprocess(self, output, original_inputs):
        """解码器后处理"""
        # original_inputs 包含原始的 decoder_inputs 字典
        image_size = original_inputs.get('image_size', (1024, 1024))
        
        return self._process_single_output(output, image_size)
    
    # def infer_decoder(self, decoder_inputs: dict):
    #     """专门为解码器设计的推理方法，支持多输入"""
    #     try:
    #         # 按照引擎中输入的顺序复制数据到对应的内存位置
    #         input_names = list(self.inputs_hdm.keys())
            
    #         for i, input_name in enumerate(input_names):
    #             if input_name in decoder_inputs:
    #                 input_data = decoder_inputs[input_name]
    #                 hdm = self.inputs_hdm[input_name]
                    
    #                 # 确保数据形状匹配
    #                 expected_shape = self.engine.get_tensor_shape(input_name)
                    
    #                 if input_data.shape != tuple(expected_shape):
    #                     print(f"Warning: Input {input_name} shape mismatch. Expected {expected_shape}, got {input_data.shape}")
                    
    #                 # 复制数据
    #                 np.copyto(hdm.host, input_data.ravel())
    #                 import pycuda.driver as cuda
    #                 cuda.memcpy_htod(hdm.device, hdm.host)
            
    #         # 执行推理
    #         if not self.context.execute_v2(bindings=self.bindings_addrs):
    #             raise RuntimeError("TensorRT execute_v2() failed")
            
    #         # 获取所有输出
    #         outputs = []
    #         for output_name in self.outputs_hdm.keys():
    #             hdm = self.outputs_hdm[output_name]
    #             import pycuda.driver as cuda
    #             cuda.memcpy_dtoh(hdm.host, hdm.device)
    #             output_shape = self.output_shapes_map[output_name]
    #             output_data = hdm.host.reshape(output_shape)
    #             outputs.append(output_data)
            
    #         return outputs
            
    #     except Exception as e:
    #         print(f"Error in decoder inference: {e}")
    #         raise
    
    def _set_dynamic_shapes(self, decoder_inputs: dict):
        
        for input_name, input_data in decoder_inputs.items():
            if input_name in self.inputs_hdm:
                # 获取引擎中定义的形状
                engine_shape = self.engine.get_tensor_shape(input_name)
                
                # 如果包含-1，说明是动态张量，需要设置实际形状
                if -1 in engine_shape:
                    actual_shape = input_data.shape
                    print(f"Setting dynamic shape for {input_name}: {actual_shape}")
                    self.context.set_input_shape(input_name, actual_shape)
    
    def _shapes_compatible(self, actual_shape, expected_shape):
        """检查实际形状和期望形状是否兼容（考虑动态维度-1）"""
        if len(actual_shape) != len(expected_shape):
            return False
        
        for actual_dim, expected_dim in zip(actual_shape, expected_shape):
            if expected_dim != -1 and actual_dim != expected_dim:
                return False
        
        return True
            
    def infer_decoder(self, decoder_inputs: dict) -> List[np.ndarray]:
        try:
            # 步骤1：首先为所有动态张量设置实际形状
            self._set_dynamic_shapes(decoder_inputs)
            
            # 步骤2：按照引擎中输入的顺序复制数据到对应的内存位置
            input_names = list(self.inputs_hdm.keys())
            
            for i, input_name in enumerate(input_names):
                if input_name in decoder_inputs:
                    input_data = decoder_inputs[input_name]
                    hdm = self.inputs_hdm[input_name]
                    
                    # 获取当前张量的实际形状（可能是动态的）
                    actual_shape = input_data.shape
                    expected_shape = self.engine.get_tensor_shape(input_name)
                    
                    print(f"Processing {input_name}: actual={actual_shape}, expected={expected_shape}")
                    
                    # 检查形状匹配（考虑动态维度）
                    if not self._shapes_compatible(actual_shape, expected_shape):
                        print(f"Warning: Input {input_name} shape mismatch. Expected {expected_shape}, got {actual_shape}")
                    
                    # 确保host内存足够大
                    required_size = np.prod(actual_shape)
                    if hdm.host.size < required_size:
                        print(f"ERROR: Host memory insufficient for {input_name}. Need {required_size}, have {hdm.host.size}")
                        continue
                    
                    # 复制数据（只复制实际需要的数据量）
                    flat_data = input_data.ravel()
                    np.copyto(hdm.host[:required_size], flat_data)
                    
                    import pycuda.driver as cuda
                    cuda.memcpy_htod(hdm.device, hdm.host[:required_size])
            
            # 步骤3：执行推理
            if not self.context.execute_v2(bindings=self.bindings_addrs):
                raise RuntimeError("TensorRT execute_v2() failed")
            
            # 步骤4：获取所有输出
            outputs = []
            for output_name in self.outputs_hdm.keys():
                hdm = self.outputs_hdm[output_name]
                import pycuda.driver as cuda
                cuda.memcpy_dtoh(hdm.host, hdm.device)
                output_shape = self.output_shapes_map[output_name]
                output_data = hdm.host.reshape(output_shape)
                outputs.append(output_data)
            
            return outputs
            
        except Exception as e:
            print(f"Error in decoder inference: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_single_output(self, outputs: List[np.ndarray], 
                             image_size: Tuple[int, int]) -> np.ndarray:
        """处理单个解码器输出"""
        masks = outputs[0]  # Shape: [1, num_masks, H, W]
        
        max_score_idx = 0
        if self.mode == "SAM2" and len(outputs) > 1:
            scores = outputs[1]  # Shape: [1, num_masks]
            # Find best mask based on score
            max_score_idx = np.argmax(scores[0])
        
        # Extract the best mask
        mask = masks[0, max_score_idx]
        
        # Convert to binary mask
        output_mask = (mask > 0).astype(np.uint8) * 255
        
        # SAM1 需要 resize，SAM2 不需要（已经是正确尺寸）
        if self.mode == "SAM":
            output_mask = cv2.resize(output_mask, image_size, interpolation=cv2.INTER_NEAREST)
        
        # Store low-res logits for next iteration
        if len(outputs) > 2:
            mask_input_size = 256 * 256
            offset_low_res = max_score_idx * mask_input_size
            low_res_logits = outputs[2].flatten()
            self.previous_masks.append(low_res_logits[offset_low_res:offset_low_res + mask_input_size].tolist())
        
        return output_mask


class Sam2TRT:
    """SAM2 TensorRT 主类，组合编码器和解码器"""
    
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.loading_model = False
        self.preprocessing = False
        self.terminating = False
        
        # Cached embeddings
        self.image_embed = None
        self.high_res_features1 = None
        self.high_res_features2 = None
        
        # Store original image size for coordinate scaling
        self.original_image_size = None
        
        # Mode setting
        self.mode = "SAM2"  # or "SAM"
    
    def __del__(self):
        if self.loading_model or self.preprocessing:
            return
        self.clear_load_model()
    
    def clear_load_model(self) -> bool:
        """清除加载的模型并释放内存"""
        try:
            self.encoder = None
            self.decoder = None
            self.image_embed = None
            self.high_res_features1 = None
            self.high_res_features2 = None
            return True
        except Exception as e:
            print(f"Error clearing model: {e}")
            return False
    
    def clear_previous_masks(self):
        """清除之前的 masks"""
        if self.decoder:
            self.decoder.previous_masks = []
    
    def terminate_preprocessing(self):
        """终止预处理"""
        self.terminating = True
    
    @staticmethod
    def model_exists(model_path: str) -> bool:
        """检查模型文件是否存在"""
        return os.path.isfile(model_path)
    
    def load_model(self, encoder_path: str, decoder_path: str) -> bool:
        """加载 SAM2 TensorRT 模型"""
        try:
            self.loading_start()
            
            if not self.clear_load_model():
                self.loading_end()
                return False
            
            if not self.model_exists(encoder_path) or not self.model_exists(decoder_path):
                print(f"Model files not found: {encoder_path} or {decoder_path}")
                self.loading_end()
                return False
            
            # 加载编码器和解码器
            self.encoder = SAM2Encoder(encoder_path)
            self.decoder = SAM2Decoder(decoder_path)
            
            # 设置模式
            self.encoder.mode = self.mode
            self.decoder.mode = self.mode
            
            if self.terminating:
                self.loading_end()
                return False
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loading_end()
            return False
        
        self.loading_end()
        return True
    
    def loading_start(self):
        """开始加载状态"""
        self.loading_model = True
    
    def loading_end(self):
        """结束加载状态"""
        self.loading_model = False
        self.terminating = False
    
    def get_input_size(self) -> Tuple[int, int]:
        """获取模型输入尺寸 (width, height)"""
        if self.encoder:
            input_shape = self.encoder.get_input_shape()
            if len(input_shape) >= 4:
                return (int(input_shape[3]), int(input_shape[2]))
        return (1024, 1024)  # Default SAM2 input size
    
    def preprocess_image(self, image: np.ndarray, original_image: np.ndarray = None) -> bool:
        """预处理图像并生成 SAM2 的 embeddings"""
        try:
            self.preprocessing_start()
            
            # Store original image size for coordinate scaling
            if original_image is not None:
                self.original_image_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
            else:
                self.original_image_size = (image.shape[1], image.shape[0])
            
            # Validate input
            expected_size = self.get_input_size()
            if image.shape[:2] != (expected_size[1], expected_size[0]):
                print(f"Expected image size {expected_size}, got {image.shape[:2]}")
                self.preprocessing_end()
                return False
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"Expected 3-channel image, got shape {image.shape}")
                self.preprocessing_end()
                return False
            
            if self.terminating:
                self.preprocessing_end()
                return False
            
            # 使用编码器进行推理
            # 由于编码器可能有多个输出，我们需要使用 TensorRT 的 infer 方法获取所有输出
            preprocessed_input = self.encoder.preprocess(image)
            
            # 直接调用编码器的 infer 方法（这只会返回主输出）
            # 我们需要获取所有输出
            outputs = self._encoder_infer_all_outputs(preprocessed_input)
            
            # 存储输出
            if self.mode == "SAM2":
                self.image_embed = outputs[0]
                if len(outputs) > 1:
                    self.high_res_features1 = outputs[1]
                if len(outputs) > 2:
                    self.high_res_features2 = outputs[2]
            else:
                self.image_embed = outputs[0]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            self.preprocessing_end()
            return False
        
        self.preprocessing_end()
        return True
    
    def _encoder_infer_all_outputs(self, input_data: np.ndarray) -> List[np.ndarray]:
        """获取编码器的所有输出"""
        # 复制输入数据
        active_input_hdm = self.encoder.inputs_hdm[self.encoder.input_name_from_engine]
        np.copyto(active_input_hdm.host, input_data.ravel())
        
        import pycuda.driver as cuda
        cuda.memcpy_htod(active_input_hdm.device, active_input_hdm.host)
        
        # 执行推理
        if not self.encoder.context.execute_v2(bindings=self.encoder.bindings_addrs):
            raise RuntimeError("TensorRT encoder execute_v2() failed")
        
        # 获取所有输出
        outputs = []
        for output_name in self.encoder.outputs_hdm.keys():
            hdm = self.encoder.outputs_hdm[output_name]
            cuda.memcpy_dtoh(hdm.host, hdm.device)
            output_shape = self.encoder.output_shapes_map[output_name]
            output_data = hdm.host.reshape(output_shape)
            outputs.append(output_data)
        
        return outputs
    
    def preprocessing_start(self):
        """开始预处理状态"""
        self.preprocessing = True
    
    def preprocessing_end(self):
        """结束预处理状态"""
        self.preprocessing = False
        self.terminating = False
    
    def _scale_points_to_model_input(self, points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """将点从原始图像坐标缩放到模型输入坐标"""
        if self.original_image_size is None:
            return [(float(x), float(y)) for x, y in points]
        
        model_input_size = self.get_input_size()
        orig_w, orig_h = self.original_image_size
        model_w, model_h = model_input_size
        
        scale_x = model_w / orig_w
        scale_y = model_h / orig_h
        
        scaled_points = []
        for x, y in points:
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            scaled_points.append((scaled_x, scaled_y))
        
        return scaled_points
    
    def set_rects_labels(self, rects: List[Tuple[int, int, int, int]], 
                        input_point_values: List[float], 
                        input_label_values: List[float]):
        """设置矩形提示 (x, y, w, h)"""
        rect_points = []
        for rect in rects:
            x, y, w, h = rect
            rect_points.extend([(x, y), (x + w, y + h)])
        
        # Scale to model input coordinates
        scaled_points = self._scale_points_to_model_input(rect_points)
        
        # Add scaled points with labels
        for i, (x, y) in enumerate(scaled_points):
            input_point_values.extend([x, y])
            if i % 2 == 0:
                input_label_values.append(2)  # Top-left corner
            else:
                input_label_values.append(3)  # Bottom-right corner
    
    def set_points_labels(self, points: List[Tuple[int, int]], label: int,
                         input_point_values: List[float], 
                         input_label_values: List[float]):
        """设置点提示 (x, y) 和标签 (1=前景, 0=背景)"""
        # Scale points to model input coordinates
        scaled_points = self._scale_points_to_model_input(points)
        
        for x, y in scaled_points:
            input_point_values.extend([x, y])
            input_label_values.append(label)
    
    def get_mask(self, input_point_values: List[float], 
                input_label_values: List[float], 
                image_size: Tuple[int, int] = None, 
                previous_mask_idx: int = -1, 
                is_next_get_mask: bool = False) -> np.ndarray:
        """获取单个 mask，支持之前的 mask 输入"""
        try:
            # Use original image size if provided, otherwise use stored size
            if image_size is None:
                if self.original_image_size is not None:
                    image_size = self.original_image_size
                else:
                    image_size = self.get_input_size()
            
            # Resize previous masks if needed
            self._resize_previous_masks(previous_mask_idx)
            
            # Prepare decoder inputs
            decoder_inputs = self._prepare_decoder_inputs(
                input_point_values, input_label_values, 1, image_size,
                previous_mask_idx, is_next_get_mask
            )
            
            # 使用解码器进行推理
            outputs = self.decoder.infer_decoder(decoder_inputs)
            
            # 添加 image_size 到输入中，供后处理使用
            decoder_inputs['image_size'] = image_size
            
            # Process outputs
            output_mask = self.decoder.postprocess(outputs, decoder_inputs)
            
            return output_mask
            
        except Exception as e:
            print(f"Error in get_mask: {e}")
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    def _resize_previous_masks(self, previous_mask_idx: int):
        """调整之前 masks 数组的大小"""
        if self.decoder and len(self.decoder.previous_masks) > previous_mask_idx + 1:
            self.decoder.previous_masks = self.decoder.previous_masks[:previous_mask_idx + 1]
    
    def _prepare_decoder_inputs(self, input_point_values: List[float], 
                              input_label_values: List[float], 
                              batch_num: int, image_size: Tuple[int, int],
                              previous_mask_idx: int = -1, 
                              is_next_get_mask: bool = False) -> dict:
        """为 SAM2 解码器准备输入"""
        inputs = {}
        
        # Get decoder input names
        decoder_input_names = list(self.decoder.inputs_hdm.keys())
        
        # Add image embeddings
        inputs[decoder_input_names[0]] = self.image_embed
        if self.mode == "SAM2":
            inputs[decoder_input_names[1]] = self.high_res_features1
            inputs[decoder_input_names[2]] = self.high_res_features2
            input_offset = 3
        else:
            input_offset = 1
        
        # Points and labels
        num_points = len(input_label_values) // batch_num
        points_array = np.array(input_point_values, dtype=np.float32).reshape(batch_num, num_points, 2)
        labels_array = np.array(input_label_values, dtype=np.float32).reshape(batch_num, num_points)
        
        inputs[decoder_input_names[input_offset]] = points_array
        inputs[decoder_input_names[input_offset + 1]] = labels_array
        
        # Mask input
        has_mask_input = 0.0
        if not is_next_get_mask and previous_mask_idx >= 0 and previous_mask_idx < len(self.decoder.previous_masks):
            mask_input = np.array(self.decoder.previous_masks[previous_mask_idx], dtype=np.float32).reshape(1, 1, 256, 256)
            has_mask_input = 1.0
        else:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        
        inputs[decoder_input_names[input_offset + 2]] = mask_input
        inputs[decoder_input_names[input_offset + 3]] = np.array([has_mask_input], dtype=np.float32)
        
        # 图像尺寸设置要与原始代码一致
        if self.mode == "SAM2":
            # SAM2模式：传递原始图像尺寸 (height, width)
            orig_im_size = np.array([image_size[1], image_size[0]], dtype=np.int64)
        else:
            # SAM1模式：传递模型输入尺寸
            model_input_size = self.get_input_size()
            orig_im_size = np.array([model_input_size[1], model_input_size[0]], dtype=np.float32)
        
        inputs[decoder_input_names[input_offset + 4]] = orig_im_size
        
        return inputs


# 使用示例
if __name__ == "__main__":
    # Initialize SAM2 TRT
    sam2 = Sam2TRT()
    
    # Load model
    encoder_path = "sam2.1_large_preprocess.engine"  # TensorRT engine file
    decoder_path = "sam2.1_large_1024x1797_python.engine"  # TensorRT engine file
    
    if sam2.load_model(encoder_path, decoder_path):
        print("SAM2 TensorRT model loaded successfully")
        
        # Load original image
        original_image = cv2.imread("Tesla.jpg")
        if original_image is not None:
            print(f"Original image size: {original_image.shape}")
            
            # Resize to model input size for preprocessing
            input_size = sam2.get_input_size()
            image_resized = cv2.resize(original_image, input_size)
            
            if sam2.preprocess_image(image_resized, original_image):
                print("Image preprocessed successfully")
                
                # 定义点击点（在原图坐标系下）
                points = [(300, 985), (850, 955)]
                input_point_values = []
                input_label_values = []
                sam2.set_points_labels(points, 1, input_point_values, input_label_values)
                
                print(f"Original points: {points}")
                print(f"Scaled points: {[(input_point_values[i], input_point_values[i+1]) for i in range(0, len(input_point_values), 2)]}")
                
                # Get mask (will be in original image size)
                mask = sam2.get_mask(input_point_values, input_label_values)
    
                
                if mask is not None and mask.size > 0:
                    print(f"Output mask size: {mask.shape}")
                    print(f"Original image size: {original_image.shape[:2]}")
                    cv2.imwrite("output_mask_trt.png", mask)
                    print("Mask saved to output_mask_trt.png")
                    
                    # Optional: Create overlay visualization
                    overlay = original_image.copy()
                    mask_colored = np.zeros_like(original_image)
                    mask_colored[mask > 0] = [0, 255, 0]  # Green mask
                    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                    cv2.imwrite("overlay_result_trt.png", overlay)
                    print("Overlay saved to overlay_result_trt.png")
                    
                    # Benchmark test
                    print("\nRunning benchmark...")
                    avg_time, _ = sam2.encoder.benchmark(image_resized, num_runs=10)
                    print(f"Encoder average inference time: {avg_time*1000:.2f} ms")
                    
            else:
                print("Failed to preprocess image")
        else:
            print("Failed to load image")
    else:
        print("Failed to load SAM2 TensorRT model")