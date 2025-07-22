# SafeTensor Model Merger

A powerful Python script for merging multiple SafeTensor models using various averaging methods including Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA).

## Features

- **Multiple Averaging Methods**: SMA, EMA, and WMA support
- **Memory Efficient**: Processes large models without excessive memory usage
- **GPU Acceleration**: CUDA support for faster processing
- **Scalar Tensor Support**: Properly handles raw float tensors and alpha weights
- **Flexible Weighting**: Custom weights for fine-tuned merging control
- **Metadata Preservation**: Stores merge parameters and source information

## Installation

```bash
# Required dependencies
pip install torch tqdm
```

## Usage

### Basic Syntax

```bash
python safetensor_merger.py <input_directory> <output_file> [OPTIONS]
```

## Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing `.safetensors` files to merge |
| `output_file` | Output path for the merged model |

### Optional Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--method` | str | `sma` | `sma`, `ema`, `wma` | Merging method to use |
| `--alpha` | float | `0.5` | `0.0 < α ≤ 1.0` | Smoothing factor for EMA |
| `--weights` | str | `None` | comma-separated | Custom weights for WMA |
| `--device` | str | `cpu` | `cpu`, `cuda` | Device for tensor operations |

## Merging Methods

### 1. Simple Moving Average (SMA)
**Formula**: `Mavg = (1/N) × Σ(Mi)`

Equal weight averaging of all models. Each model contributes equally to the final result.

```bash
python safetensor_merger.py ./models ./merged_sma.safetensors --method sma
```

**Use Case**: When all models are equally important and you want a balanced merge.

### 2. Exponential Moving Average (EMA)
**Formula**: `EMA(t) = α × Current + (1-α) × EMA(t-1)`

Exponentially weighted average where recent models have more influence.

```bash
# Higher alpha = more weight to recent models
python safetensor_merger.py ./models ./merged_ema.safetensors --method ema --alpha 0.7

# Lower alpha = more weight to earlier models  
python safetensor_merger.py ./models ./merged_ema.safetensors --method ema --alpha 0.3
```

**Alpha Parameter Guide**:
- `α = 0.1-0.3`: Heavily favor earlier models
- `α = 0.4-0.6`: Balanced weighting
- `α = 0.7-0.9`: Heavily favor recent models

**Use Case**: When model order matters and you want recent models to have more influence.

### 3. Weighted Moving Average (WMA)
**Formula**: `WMA = Σ(Wi × Mi) / Σ(Wi)`

Custom weighted average with user-defined weights.

```bash
# Custom weights (must match number of models)
python safetensor_merger.py ./models ./merged_wma.safetensors --method wma --weights "0.5,0.3,0.2"

# Default linear decreasing weights [N, N-1, ..., 2, 1]
python safetensor_merger.py ./models ./merged_wma.safetensors --method wma
```

**Weight Examples**:
- `"1,1,1"`: Equal weights (same as SMA)
- `"0.6,0.3,0.1"`: Heavily favor first model
- `"0.1,0.3,0.6"`: Heavily favor last model
- `"2,1,1"`: Double weight to first model

**Use Case**: When you know specific models should have different importance levels.

## Examples

### Basic Examples

```bash
# Simple merge with equal weights
python safetensor_merger.py ./my_models ./output.safetensors

# Use GPU acceleration
python safetensor_merger.py ./my_models ./output.safetensors --device cuda
```

### EMA Examples

```bash
# Conservative EMA (favor earlier models)
python safetensor_merger.py ./models ./conservative_merge.safetensors --method ema --alpha 0.2

# Aggressive EMA (favor recent models)
python safetensor_merger.py ./models ./aggressive_merge.safetensors --method ema --alpha 0.8

# Balanced EMA
python safetensor_merger.py ./models ./balanced_merge.safetensors --method ema --alpha 0.5
```

### WMA Examples

```bash
# Equal importance to all 3 models
python safetensor_merger.py ./models ./equal_merge.safetensors --method wma --weights "1,1,1"

# Pyramid weighting (decreasing importance)
python safetensor_merger.py ./models ./pyramid_merge.safetensors --method wma --weights "0.5,0.3,0.2"

# Focus on middle model
python safetensor_merger.py ./models ./middle_focus.safetensors --method wma --weights "0.2,0.6,0.2"

# Binary choice (ignore middle model)
python safetensor_merger.py ./models ./binary_merge.safetensors --method wma --weights "0.5,0,0.5"
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple model directories
for dir in model_*; do
  python safetensor_merger.py "$dir" "merged_${dir}.safetensors" --method ema --alpha 0.6
done
```

### Memory Optimization

```bash
# For large models, use CPU to avoid GPU memory issues
python safetensor_merger.py ./large_models ./merged.safetensors --device cpu --method sma
```

## Output Information

The script provides detailed output including:

- **Model Discovery**: Lists all found `.safetensors` files
- **Common Tensors**: Reports number of tensors present in all models
- **Method Parameters**: Shows selected method and parameters
- **Progress**: Real-time progress bar during processing
- **Metadata**: Stores merge information in the output file

### Sample Output

```
Merging 3 models using EMA (α=0.6)...
Model files:
  1. model_1.safetensors
  2. model_2.safetensors  
  3. model_3.safetensors

Finding common tensor keys...
Found 145 common tensors

Computing EMA tensors...
Processing tensors: 100%|████████| 145/145 [00:23<00:00, 6.21it/s]

Using memory efficient save file: ./merged_ema.safetensors

✅ Successfully merged 3 models using EMA
📁 Output saved to: ./merged_ema.safetensors
```

## Metadata Storage

Each merged model contains metadata with:

- `merged_models_count`: Number of source models
- `merge_method`: Method used (SMA/EMA/WMA)  
- `source_files`: List of source filenames
- Method-specific parameters (`alpha` for EMA, `weights` for WMA)

## Error Handling

Common errors and solutions:

### No Models Found
```
Error: No .safetensors files found in ./models
```
**Solution**: Ensure the directory contains `.safetensors` files

### Weight Count Mismatch  
```
Error: Number of weights (2) must match number of models (3)
```
**Solution**: Provide correct number of comma-separated weights

### Invalid Alpha
```
Error: Alpha for EMA must be between 0 and 1 (exclusive of 0)
```
**Solution**: Use alpha value in range (0.0, 1.0]

### CUDA Unavailable
```
Warning: CUDA requested but not available. Using CPU instead.
```
**Solution**: Install CUDA-enabled PyTorch or use `--device cpu`

## Performance Tips

1. **Use GPU**: Add `--device cuda` for faster processing on compatible hardware
2. **Method Selection**: SMA is fastest, WMA is most flexible, EMA is good for sequential importance
3. **Memory**: For very large models, stick with CPU to avoid memory issues
4. **Weights**: Pre-calculate optimal weights for WMA based on model performance

## Supported Tensor Types

The script handles all standard tensor types including:
- Standard tensors (F32, F16, BF16, etc.)
- Integer tensors (I8, I16, I32, I64)
- Boolean tensors
- **Scalar tensors** (alpha weights, bias terms, etc.)
- Float8 types (if supported by PyTorch version)

## Requirements

- Tested on Python 3.11.9
- PyTorch 1.9+
- tqdm
- Standard library modules (json, struct, pathlib, argparse)

## License

This script is provided as-is for educational and research purposes.
