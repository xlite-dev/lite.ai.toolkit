import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Optional
import os

class Sam2:
    def __init__(self):
        self.session_encoder = None
        self.session_decoder = None
        self.loading_model = False
        self.preprocessing = False
        self.terminating = False
        
        # Model shapes and outputs
        self.input_shape_encoder = []
        self.output_shape_encoder = []
        self.high_res_features1_shape = []
        self.high_res_features2_shape = []
        
        # Cached embeddings
        self.output_tensor_values_encoder = None
        self.high_res_features1 = None
        self.high_res_features2 = None
        
        # Previous masks for iterative refinement
        self.previous_masks = []
        
        # ONNX Runtime providers
        self.providers = ['CPUExecutionProvider']
        
        # Store original image size for coordinate scaling
        self.original_image_size = None
        
        # Mode setting
        self.mode = "SAM2"  # or "SAM"
        
    def __del__(self):
        if self.loading_model or self.preprocessing:
            return
        self.clear_load_model()
        self.clear_previous_masks()
    
    def clear_load_model(self) -> bool:
        """Clear loaded models and free memory"""
        try:
            self.session_encoder = None
            self.session_decoder = None
            self.input_shape_encoder = []
            self.output_shape_encoder = []
            self.high_res_features1_shape = []
            self.high_res_features2_shape = []
            self.output_tensor_values_encoder = None
            self.high_res_features1 = None
            self.high_res_features2 = None
            return True
        except Exception as e:
            print(f"Error clearing model: {e}")
            return False
    
    def clear_previous_masks(self):
        """Clear previous masks"""
        self.previous_masks = []
    
    def resize_previous_masks(self, previous_mask_idx: int):
        """Resize previous masks array"""
        if len(self.previous_masks) > previous_mask_idx + 1:
            self.previous_masks = self.previous_masks[:previous_mask_idx + 1]
    
    def terminate_preprocessing(self):
        """Terminate preprocessing"""
        self.terminating = True
    
    @staticmethod
    def model_exists(model_path: str) -> bool:
        """Check if model file exists"""
        return os.path.isfile(model_path)
    
    def load_model(self, encoder_path: str, decoder_path: str, 
                   threads_number: int = 1, device: str = "cpu") -> bool:
        """Load SAM2 ONNX models"""
        try:
            self.loading_start()
            
            if not self.clear_load_model():
                self.loading_end()
                return False
            
            if not self.model_exists(encoder_path) or not self.model_exists(decoder_path):
                print(f"Model files not found: {encoder_path} or {decoder_path}")
                self.loading_end()
                return False
            
            # Setup providers based on device
            if device == "cpu":
                self.providers = ['CPUExecutionProvider']
            elif device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1]) if ":" in device else 0
                self.providers = [
                    ('CUDAExecutionProvider', {'device_id': gpu_id}),
                    'CPUExecutionProvider'
                ]
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads_number
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Load sessions
            self.session_encoder = ort.InferenceSession(
                encoder_path, sess_options, providers=self.providers
            )
            self.session_decoder = ort.InferenceSession(
                decoder_path, sess_options, providers=self.providers
            )
            
            # Get input/output shapes
            encoder_inputs = self.session_encoder.get_inputs()
            encoder_outputs = self.session_encoder.get_outputs()
            
            self.input_shape_encoder = encoder_inputs[0].shape
            self.output_shape_encoder = encoder_outputs[0].shape
            
            if self.mode == "SAM2":
                self.high_res_features1_shape = encoder_outputs[1].shape
                self.high_res_features2_shape = encoder_outputs[2].shape
            
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
        """Start loading state"""
        self.loading_model = True
    
    def loading_end(self):
        """End loading state"""
        self.loading_model = False
        self.terminating = False
    
    def get_input_size(self) -> Tuple[int, int]:
        """Get model input size (width, height)"""
        if len(self.input_shape_encoder) >= 4:
            return (int(self.input_shape_encoder[3]), int(self.input_shape_encoder[2]))
        return (1024, 1024)  # Default SAM2 input size
    
    def preprocess_image(self, image: np.ndarray, original_image: np.ndarray = None) -> bool:
        """Preprocess image and generate embeddings for SAM2"""
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
            
            # Prepare input tensor based on mode
            if self.mode == "SAM2":
                input_tensor = self._prepare_float_input(image)
            else:
                input_tensor = self._prepare_uint8_input(image)
            
            if self.terminating:
                self.preprocessing_end()
                return False
            
            # Run encoder
            encoder_inputs = self.session_encoder.get_inputs()
            encoder_outputs = self.session_encoder.get_outputs()
            
            input_name = encoder_inputs[0].name
            output_names = [output.name for output in encoder_outputs]
            
            outputs = self.session_encoder.run(output_names, {input_name: input_tensor})
            
            # Store outputs
            self.output_tensor_values_encoder = outputs[0]
            if self.mode == "SAM2":
                self.high_res_features1 = outputs[1]
                self.high_res_features2 = outputs[2]
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            self.preprocessing_end()
            return False
        
        self.preprocessing_end()
        return True
    
    def _prepare_float_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare float input tensor for SAM2 - 对应C++中的float处理"""
        # ImageNet normalization for SAM2
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Convert BGR to RGB and normalize - 对应C++中的通道顺序处理
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
        """Prepare uint8 input tensor for SAM1 - 对应C++中的uint8处理"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to NCHW format
        image_tensor = np.transpose(image_rgb, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)
        
        return image_tensor.astype(np.uint8)
    
    def preprocessing_start(self):
        """Start preprocessing state"""
        self.preprocessing = True
    
    def preprocessing_end(self):
        """End preprocessing state"""
        self.preprocessing = False
        self.terminating = False
    
    def _scale_points_to_model_input(self, points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Scale points from original image coordinates to model input coordinates"""
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
        """Set rectangle prompts (x, y, w, h)"""
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
        """Set point prompts (x, y) with label (1=foreground, 0=background)"""
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
        """Get single mask with optional previous mask input - 对应C++的getMask函数"""
        try:
            # Use original image size if provided, otherwise use stored size
            if image_size is None:
                if self.original_image_size is not None:
                    image_size = self.original_image_size
                else:
                    image_size = self.get_input_size()
            
            # Resize previous masks if needed
            self.resize_previous_masks(previous_mask_idx)
            
            # Prepare decoder inputs
            decoder_inputs = self._prepare_decoder_inputs(
                input_point_values, input_label_values, 1, image_size,
                previous_mask_idx, is_next_get_mask
            )
            
            # Run decoder
            decoder_input_names = [inp.name for inp in self.session_decoder.get_inputs()]
            decoder_output_names = [out.name for out in self.session_decoder.get_outputs()]
            
            outputs = self.session_decoder.run(decoder_output_names, decoder_inputs)
            
            # Process outputs and store previous mask
            output_mask = self._process_single_output(outputs, image_size)
            
            return output_mask
            
        except Exception as e:
            print(f"Error in get_mask: {e}")
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    def _prepare_decoder_inputs(self, input_point_values: List[float], 
                              input_label_values: List[float], 
                              batch_num: int, image_size: Tuple[int, int],
                              previous_mask_idx: int = -1, 
                              is_next_get_mask: bool = False) -> dict:
        """Prepare inputs for SAM2 decoder - 对应C++中的多个set函数"""
        inputs = {}
        
        # Get decoder input names
        decoder_input_names = [inp.name for inp in self.session_decoder.get_inputs()]
        
        # Add image embeddings
        inputs[decoder_input_names[0]] = self.output_tensor_values_encoder
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
        mask_input_size = 256 * 256
        has_mask_input = 0.0
        if not is_next_get_mask and previous_mask_idx >= 0 and previous_mask_idx < len(self.previous_masks):
            mask_input = np.array(self.previous_masks[previous_mask_idx], dtype=np.float32).reshape(1, 1, 256, 256)
            has_mask_input = 1.0
        else:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        
        inputs[decoder_input_names[input_offset + 2]] = mask_input
        inputs[decoder_input_names[input_offset + 3]] = np.array([has_mask_input], dtype=np.float32)
        
        # 关键修正：图像尺寸设置要与C++一致
        if self.mode == "SAM2":
            # SAM2模式：传递原始图像尺寸 (height, width)
            orig_im_size = np.array([image_size[1], image_size[0]], dtype=np.int64)
        else:
            # SAM1模式：传递模型输入尺寸
            model_input_size = self.get_input_size()
            orig_im_size = np.array([model_input_size[1], model_input_size[0]], dtype=np.float32)
        
        inputs[decoder_input_names[input_offset + 4]] = orig_im_size
        
        return inputs
    
    def _process_single_output(self, outputs: List[np.ndarray], 
                             image_size: Tuple[int, int]) -> np.ndarray:
        """Process single decoder output - 对应C++中的mask处理逻辑"""
        masks = outputs[0]  # Shape: [1, num_masks, H, W]
        
        max_score_idx = 0
        if self.mode == "SAM2":
            scores = outputs[1]  # Shape: [1, num_masks]
            # Find best mask based on score
            max_score_idx = np.argmax(scores[0])
        
        # Extract the best mask
        mask = masks[0, max_score_idx]
        
        # Convert to binary mask
        output_mask = (mask > 0).astype(np.uint8) * 255
        
        # 关键：SAM1需要resize，SAM2不需要（已经是正确尺寸）
        if self.mode == "SAM":
            output_mask = cv2.resize(output_mask, image_size, interpolation=cv2.INTER_NEAREST)
        
        # Store low-res logits for next iteration
        if len(outputs) > 2:
            mask_input_size = 256 * 256
            offset_low_res = max_score_idx * mask_input_size
            low_res_logits = outputs[2].flatten()
            self.previous_masks.append(low_res_logits[offset_low_res:offset_low_res + mask_input_size].tolist())
        
        return output_mask

# 使用示例
if __name__ == "__main__":
    # Initialize SAM2
    sam2 = Sam2()
    
    # Load model
    encoder_path = "sam2.1_large_preprocess.onnx"
    decoder_path = "sam2.1_large.onnx"
    
    if sam2.load_model(encoder_path, decoder_path, device="cpu"):
        print("SAM2 model loaded successfully")
        
        # Load original image
        original_image = cv2.imread("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
        if original_image is not None:
            print(f"Original image size: {original_image.shape}")
            
            # Resize to model input size for preprocessing
            input_size = sam2.get_input_size()
            image_resized = cv2.resize(original_image, input_size)
            
            if sam2.preprocess_image(image_resized, original_image):
                print("Image preprocessed successfully")
                
                # 定义点击点（在原图坐标系下）- 根据你的斑马图片调整坐标
                points = [(647, 271),(984,250),(1426,278),(236,265)]  # 尝试点击斑马身体
                input_point_values = []
                input_label_values = []
                sam2.set_points_labels(points, 1, input_point_values, input_label_values)  # 1 = foreground
                
                print(f"Original points: {points}")
                print(f"Scaled points: {[(input_point_values[i], input_point_values[i+1]) for i in range(0, len(input_point_values), 2)]}")
                
                # Get mask (will be in original image size)
                mask = sam2.get_mask(input_point_values, input_label_values)
                
                if mask is not None and mask.size > 0:
                    print(f"Output mask size: {mask.shape}")
                    print(f"Original image size: {original_image.shape[:2]}")
                    cv2.imwrite("output_mask_fixed.png", mask)
                    print("Mask saved to output_mask_fixed.png")
                    
                    # Optional: Create overlay visualization
                    overlay = original_image.copy()
                    mask_colored = np.zeros_like(original_image)
                    mask_colored[mask > 0] = [0, 255, 0]  # Green mask
                    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                    cv2.imwrite("overlay_result_fixed.png", overlay)
                    print("Overlay saved to overlay_result_fixed.png")
            else:
                print("Failed to preprocess image")
        else:
            print("Failed to load image")
    else:
        print("Failed to load SAM2 model")