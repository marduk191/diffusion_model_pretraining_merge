#!/usr/bin/env python3
"""
SafeTensor Model Merger with Multiple Averaging Methods
Supports Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA)
https://github.com/marduk191
"""

import os
import sys
import json
import struct
import torch
from pathlib import Path
from typing import Dict, List, Any
import argparse
from tqdm import tqdm
import math

# Import your custom classes (assumes they're in the same directory or in PYTHONPATH)
# You may need to adjust these imports based on your file structure
class MemoryEfficientSafeOpen:
    # does not support metadata loading
    def __init__(self, filename):
        self.filename = filename
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            return torch.empty(shape, dtype=dtype)
        
        tensor_bytes = bytearray(tensor_bytes)  # make it writable
        byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        tensor = byte_tensor.view(dtype).reshape(shape)
        
        # Handle scalar tensors (0-dimensional tensors)
        if len(shape) == 0:
            return tensor.squeeze()
        
        return tensor

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        # Handle scalar tensors (0-dimensional)
        if v.numel() == 0 or (v.dim() == 0):
            if v.dim() == 0:
                # Scalar tensor - store as single element
                size = v.element_size()
                header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
                offset += size
            else:
                # Empty tensor
                header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            # Skip truly empty tensors (numel == 0 but not scalar)
            if v.numel() == 0 and v.dim() > 0:
                continue
                
            # Handle scalar tensors and regular tensors
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # scalar tensor
                        # Create a temporary view for scalar
                        scalar_bytes = v.detach().view(torch.uint8)
                        scalar_bytes.cpu().numpy().tofile(f)
                    else:
                        tensor_bytes = v.contiguous().view(torch.uint8)
                        tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # scalar tensor
                    # Handle scalar tensors properly
                    scalar_bytes = v.detach().view(torch.uint8)
                    scalar_bytes.numpy().tofile(f)
                else:
                    v.contiguous().view(torch.uint8).numpy().tofile(f)


def find_safetensor_files(directory: str) -> List[str]:
    """Find all .safetensors files in the given directory."""
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    safetensor_files = list(directory.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {directory}")
    
    return [str(f) for f in sorted(safetensor_files)]


def get_common_keys(model_files: List[str]) -> List[str]:
    """Get the common tensor keys across all model files."""
    if not model_files:
        return []
    
    # Get keys from the first model
    with MemoryEfficientSafeOpen(model_files[0]) as f:
        common_keys = set(f.keys())
    
    # Find intersection with all other models
    for model_file in model_files[1:]:
        with MemoryEfficientSafeOpen(model_file) as f:
            model_keys = set(f.keys())
            common_keys = common_keys.intersection(model_keys)
    
    return sorted(list(common_keys))


def safe_tensor_operation(tensor1, tensor2, operation):
    """
    Safely perform operations on tensors, handling scalars and different shapes.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor  
        operation: Function to apply (e.g., torch.add, torch.mul)
    
    Returns:
        Result tensor
    """
    # Handle scalar tensors
    if tensor1.dim() == 0 and tensor2.dim() == 0:
        # Both scalars
        return operation(tensor1, tensor2)
    elif tensor1.dim() == 0:
        # tensor1 is scalar, tensor2 is not
        return operation(tensor1.expand_as(tensor2), tensor2)
    elif tensor2.dim() == 0:
        # tensor2 is scalar, tensor1 is not  
        return operation(tensor1, tensor2.expand_as(tensor1))
    else:
        # Both are regular tensors
        if tensor1.shape != tensor2.shape:
            raise ValueError(f"Tensor shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return operation(tensor1, tensor2)


def merge_models_sma(model_files: List[str], output_path: str, device: str = "cpu"):
    """
    Merge multiple safetensor models using Simple Moving Average (SMA).
    Formula: Mavg = (1/N) * Σ(Mi) where N is the number of models
    """
    if not model_files:
        raise ValueError("No model files provided")
    
    n_models = len(model_files)
    print(f"Merging {n_models} models using SMA...")
    
    # Get common tensor keys
    common_keys = get_common_keys(model_files)
    
    if not common_keys:
        raise ValueError("No common tensor keys found across all models")
    
    # Initialize averaged tensors dictionary
    averaged_tensors = {}
    
    # Process each tensor key
    print("\nAveraging tensors...")
    for key in tqdm(common_keys, desc="Processing tensors"):
        tensor_sum = None
        
        # Sum tensors across all models
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file) as f:
                tensor = f.get_tensor(key).to(device, non_blocking=True)
                
                if tensor_sum is None:
                    tensor_sum = tensor.clone()
                else:
                    tensor_sum = safe_tensor_operation(tensor_sum, tensor, torch.add)
        
        # Apply SMA formula: Mavg = (1/N) * Σ(Mi)
        if tensor_sum.dim() == 0:
            # Handle scalar division
            averaged_tensors[key] = tensor_sum / n_models
        else:
            averaged_tensors[key] = tensor_sum / n_models
    
    # Prepare metadata
    metadata = {
        "merged_models_count": str(n_models),
        "merge_method": "SMA",
        "source_files": json.dumps([Path(f).name for f in model_files])
    }
    
    # Save the averaged model
    print(f"\nSaving merged model to: {output_path}")
    mem_eff_save_file(averaged_tensors, output_path, metadata)
    
    return output_path


def merge_models_ema(model_files: List[str], output_path: str, alpha: float = 0.5, device: str = "cpu"):
    """
    Merge multiple safetensor models using Exponential Moving Average (EMA).
    Formula: EMA(t) = α * Current + (1-α) * EMA(t-1)
    where α is the smoothing factor (0 < α ≤ 1)
    """
    if not model_files:
        raise ValueError("No model files provided")
    
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1 (exclusive of 0, inclusive of 1)")
    
    n_models = len(model_files)
    print(f"Merging {n_models} models using EMA (α={alpha})...")
    
    # Get common tensor keys
    common_keys = get_common_keys(model_files)
    
    if not common_keys:
        raise ValueError("No common tensor keys found across all models")
    
    # Initialize EMA tensors dictionary
    ema_tensors = {}
    
    # Process each tensor key
    print("\nComputing EMA tensors...")
    for key in tqdm(common_keys, desc="Processing tensors"):
        ema_tensor = None
        
        # Apply EMA formula iteratively
        for i, model_file in enumerate(model_files):
            with MemoryEfficientSafeOpen(model_file) as f:
                current_tensor = f.get_tensor(key).to(device, non_blocking=True)
                
                if ema_tensor is None:
                    # Initialize with first tensor
                    ema_tensor = current_tensor.clone()
                else:
                    # EMA formula: EMA(t) = α * Current + (1-α) * EMA(t-1)
                    if current_tensor.dim() == 0 and ema_tensor.dim() == 0:
                        # Both scalars
                        ema_tensor = alpha * current_tensor + (1 - alpha) * ema_tensor
                    elif current_tensor.dim() == 0:
                        # current is scalar, ema is not (shouldn't happen with same keys)
                        raise ValueError(f"Tensor dimension mismatch for key '{key}': scalar vs tensor")
                    elif ema_tensor.dim() == 0:
                        # ema is scalar, current is not (shouldn't happen with same keys)
                        raise ValueError(f"Tensor dimension mismatch for key '{key}': tensor vs scalar")
                    else:
                        # Both regular tensors
                        ema_tensor = alpha * current_tensor + (1 - alpha) * ema_tensor
        
        ema_tensors[key] = ema_tensor
    
    # Prepare metadata
    metadata = {
        "merged_models_count": str(n_models),
        "merge_method": "EMA",
        "alpha": str(alpha),
        "source_files": json.dumps([Path(f).name for f in model_files])
    }
    
    # Save the EMA model
    print(f"\nSaving merged model to: {output_path}")
    mem_eff_save_file(ema_tensors, output_path, metadata)
    
    return output_path


def merge_models_wma(model_files: List[str], output_path: str, weights: List[float] = None, device: str = "cpu"):
    """
    Merge multiple safetensor models using Weighted Moving Average (WMA).
    Formula: WMA = Σ(Wi * Mi) / Σ(Wi) where Wi are weights and Mi are models
    If no weights provided, uses linear decreasing weights (most recent model gets highest weight)
    """
    if not model_files:
        raise ValueError("No model files provided")
    
    n_models = len(model_files)
    
    # Generate default weights if none provided (linear decreasing: n, n-1, ..., 1)
    if weights is None:
        weights = list(range(n_models, 0, -1))  # [n, n-1, ..., 2, 1]
        print(f"Using default linear decreasing weights: {weights}")
    else:
        if len(weights) != n_models:
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({n_models})")
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive")
    
    # Normalize weights
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    print(f"Merging {n_models} models using WMA...")
    print(f"Weights: {weights}")
    print(f"Normalized weights: {[f'{w:.4f}' for w in normalized_weights]}")
    
    # Get common tensor keys
    common_keys = get_common_keys(model_files)
    
    if not common_keys:
        raise ValueError("No common tensor keys found across all models")
    
    # Initialize weighted tensors dictionary
    weighted_tensors = {}
    
    # Process each tensor key
    print("\nComputing weighted average tensors...")
    for key in tqdm(common_keys, desc="Processing tensors"):
        weighted_sum = None
        
        # Compute weighted sum
        for i, model_file in enumerate(model_files):
            with MemoryEfficientSafeOpen(model_file) as f:
                tensor = f.get_tensor(key).to(device, non_blocking=True)
                
                # Apply weight
                if tensor.dim() == 0:
                    # Handle scalar tensors
                    weighted_tensor = tensor * normalized_weights[i]
                else:
                    weighted_tensor = tensor * normalized_weights[i]
                
                if weighted_sum is None:
                    weighted_sum = weighted_tensor
                else:
                    weighted_sum = safe_tensor_operation(weighted_sum, weighted_tensor, torch.add)
        
        weighted_tensors[key] = weighted_sum
    
    # Prepare metadata
    metadata = {
        "merged_models_count": str(n_models),
        "merge_method": "WMA",
        "weights": json.dumps(weights),
        "normalized_weights": json.dumps([round(w, 6) for w in normalized_weights]),
        "source_files": json.dumps([Path(f).name for f in model_files])
    }
    
    # Save the weighted model
    print(f"\nSaving merged model to: {output_path}")
    mem_eff_save_file(weighted_tensors, output_path, metadata)
    
    return output_path


def merge_models(model_files: List[str], output_path: str, method: str = "sma", 
                alpha: float = 0.5, weights: List[float] = None, device: str = "cpu"):
    """
    Merge models using the specified method.
    
    Args:
        model_files: List of model file paths
        output_path: Output file path
        method: Merging method ('sma', 'ema', 'wma')
        alpha: Smoothing factor for EMA (0 < α ≤ 1)
        weights: Custom weights for WMA
        device: Device to use for computations
    """
    print("Model files:")
    for i, file in enumerate(model_files, 1):
        print(f"  {i}. {Path(file).name}")
    
    print(f"\nFinding common tensor keys...")
    common_keys = get_common_keys(model_files)
    print(f"Found {len(common_keys)} common tensors")
    
    if method.lower() == "sma":
        return merge_models_sma(model_files, output_path, device)
    elif method.lower() == "ema":
        return merge_models_ema(model_files, output_path, alpha, device)
    elif method.lower() == "wma":
        return merge_models_wma(model_files, output_path, weights, device)
    else:
        raise ValueError(f"Unknown merging method: {method}")


def parse_weights(weights_str: str) -> List[float]:
    """Parse comma-separated weights string into list of floats."""
    if not weights_str:
        return None
    
    try:
        weights = [float(w.strip()) for w in weights_str.split(",")]
        return weights
    except ValueError as e:
        raise ValueError(f"Invalid weights format. Use comma-separated numbers: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge SafeTensor models using various averaging methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Merging Methods:
  SMA (Simple Moving Average): Equal weight average of all models
  EMA (Exponential Moving Average): Exponentially weighted average with smoothing factor α
  WMA (Weighted Moving Average): Custom weighted average

Examples:
  # Simple Moving Average (default)
  python merge_safetensors.py ./models ./merged_sma.safetensors
  
  # Exponential Moving Average with α=0.3
  python merge_safetensors.py ./models ./merged_ema.safetensors --method ema --alpha 0.3
  
  # Weighted Moving Average with custom weights
  python merge_safetensors.py ./models ./merged_wma.safetensors --method wma --weights "0.5,0.3,0.2"
  
  # Use GPU acceleration
  python merge_safetensors.py ./models ./merged.safetensors --device cuda
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing .safetensors files to merge"
    )
    
    parser.add_argument(
        "output_file",
        help="Output path for the merged model"
    )
    
    parser.add_argument(
        "--method",
        default="sma",
        choices=["sma", "ema", "wma"],
        help="Merging method: sma (Simple), ema (Exponential), wma (Weighted) (default: sma)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Smoothing factor for EMA (0 < α ≤ 1, default: 0.5)"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated weights for WMA (e.g., '0.5,0.3,0.2'). If not specified for WMA, uses linear decreasing weights."
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for tensor operations (default: cpu)"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate CUDA availability if requested
        if args.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU instead.")
            args.device = "cpu"
        
        # Parse weights if provided
        weights = parse_weights(args.weights) if args.weights else None
        
        # Find model files
        model_files = find_safetensor_files(args.input_dir)
        
        if len(model_files) < 2:
            print("Warning: Only one model found. Nothing to merge.")
            return
        
        # Validate method-specific parameters
        if args.method == "ema" and not (0 < args.alpha <= 1):
            print("Error: Alpha for EMA must be between 0 and 1 (exclusive of 0)")
            sys.exit(1)
        
        if args.method == "wma" and weights and len(weights) != len(model_files):
            print(f"Error: Number of weights ({len(weights)}) must match number of models ({len(model_files)})")
            sys.exit(1)
        
        # Merge models
        output_path = merge_models(
            model_files=model_files,
            output_path=args.output_file,
            method=args.method,
            alpha=args.alpha,
            weights=weights,
            device=args.device
        )
        
        print(f"\n✅ Successfully merged {len(model_files)} models using {args.method.upper()}")
        print(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()