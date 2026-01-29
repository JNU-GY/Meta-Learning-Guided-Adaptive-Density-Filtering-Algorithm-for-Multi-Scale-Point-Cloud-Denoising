# Meta-learning Guided Density Filtering for Robust Multi-Scale Point Cloud Denoising

A sophisticated point cloud denoising system that combines density-based outlier removal with advanced machine learning optimization techniques.

## Overview

Point cloud data, widely used in 3D reconstruction and industrial inspection, often suffer from multi-scale noise and outliers, compromising data quality and feature preservation. To address this, we propose a meta-learning guided adaptive density filtering algorithm. Our approach integrates multi-scale hybrid sampling with a momentum-based gradient descent meta-learning framework to dynamically optimize neighbourhood parameters for local density estimation. A density-adaptive filter, leveraging Gaussian kernel density and interquartile range analysis, hierarchically removes noise while preserving geometric features. Experimental results demonstrate a Recall metric consistently above 0.98 and a reduction in Chamfer distance by up to 80.9% compared to traditional methods, showcasing robustness in multi-scale noise removal for real-world applications such as industrial inspection.
<img width="921" height="219" alt="image" src="https://github.com/user-attachments/assets/ceec33f6-11f5-4db6-ab28-a856653e2a5d" />


## Features

- **Automatic Parameter Optimization**: Uses meta-learning techniques to automatically determine optimal denoising parameters
- **Multi-scale Sampling**: Employs intelligent sampling strategies for efficient processing of large point clouds
- **Density-based Filtering**: Removes outliers based on local point density analysis
- **PLY Format Support**: Full support for PLY (Polygon File Format) input and output
- **Batch Processing**: Processes multiple point cloud files in a single run
- **Real-time Feedback**: Provides detailed processing statistics and progress information

## Core Algorithm

### Architecture

The denoising pipeline consists of three main components:

1. **Multi-scale Hybrid Sampling Module**
2. **Meta-learning Parameter Optimizer**
3. **Density-based Outlier Removal**

### Multi-scale Sampling

The algorithm employs a three-tier sampling strategy:

```python
ratios = [0.4, 0.3, 0.3]  # Global: 40%, Mid-scale: 30%, Local: 30%
```

- **Global Sampling**: Random sampling across the entire point cloud
- **Mid-scale Sampling**: Voxel-based downsampling to capture medium-scale features
- **Local Sampling**: Focused sampling of remaining regions

### Meta-learning Optimization

The system uses a sophisticated meta-learning approach to optimize the critical parameter `k_neighbors`:

- **Inner Loop**: Gradient descent optimization for individual point clouds
- **Outer Loop**: Meta-learning across multiple samples to learn optimal initialization
- **Adaptive Learning Rate**: Momentum-based optimization with early stopping

### Density-based Outlier Removal

The core filtering algorithm uses statistical density analysis:

1. **Density Calculation**: Computes local point density using k-nearest neighbors
2. **Gaussian Kernel**: Optional Gaussian weighting for smoother density estimation
3. **IQR-based Thresholding**: Uses interquartile range (IQR) for robust outlier detection
4. **Adaptive Threshold**: `threshold = median(density) - iqr_ratio × (Q3 - Q1)`

## Dependencies

### Required Packages

```bash
pip install numpy
pip install open3d
pip install scikit-learn
```

### System Requirements

- **Python**: 3.7+
- **Memory**: 8GB+ RAM recommended for large point clouds
- **Storage**: Sufficient space for input/output PLY files

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install numpy open3d scikit-learn
   ```
3. Place your PLY files in the `input/` directory

## Usage

### GUI Interface (Recommended)

The project provides a user-friendly graphical interface:

```bash
pip install -r requirements_gui.txt
python gui.py
```

GUI Features:
- **Visualization**: Real-time display of original point cloud (blue) and denoised point cloud (red)
- **Parameter Adjustment**: Slider to adjust IQR parameter (1.0-5.0)
- **File Selection**: Convenient input/output file selection dialogs
- **Real-time Logging**: Detailed processing progress and statistics display

### Basic Usage (Command Line)

```python
from main import main

# Process all PLY files in input folder
main(
    input_folder='input',
    output_folder='output',
    iqr_ratio=2.0  # Adjust denoising sensitivity (higher = more aggressive)
)
```

### Command Line Execution

```bash
python main.py
```

The script will:
1. Scan the `input/` folder for `.ply` files
2. Process each file with optimized parameters
3. Save denoised results to `output/` folder with `denoised_` prefix

### Parameter Tuning

- **iqr_ratio**: Controls denoising aggressiveness
  - Lower values (1.0-1.5): Conservative denoising, preserves more points
  - Higher values (2.0-3.0): Aggressive denoising, removes more outliers
  - Default: 2.0

## Algorithm Details

### Density Calculation

```python
def calculate_point_density(self, points):
    nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, n_jobs=-1).fit(points)
    distances, _ = nn.kneighbors(points)

    if self.use_gaussian:
        sigma = np.mean(distances[:, 1:])
        return np.sum(np.exp(-(distances[:, 1:] ** 2) / (2 * sigma ** 2)), axis=1)
    else:
        return 1 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
```

### Evaluation Function

The system uses a composite evaluation metric balancing density uniformity and geometric preservation:

```python
score = -(0.9 × density_std + 0.1 × geometry_score)
```

- **Density Uniformity** (90% weight): Measures how evenly distributed the cleaned points are
- **Geometric Preservation** (10% weight): Ensures important geometric features are retained

### Meta-learning Process

1. **Sample Generation**: Create representative samples using multi-scale sampling
2. **Inner Optimization**: Optimize k parameter for each sample using gradient descent
3. **Meta-gradient Computation**: Calculate how parameter changes affect performance
4. **Parameter Update**: Update meta-parameters using computed gradients

## File Structure

```
├── main.py                 # Main processing script
├── densityV3.py           # Core algorithm implementation
├── input/                 # Input PLY files directory
├── output/                # Processed output directory
├── README.md             # This documentation
└── manuscript file.md    # Technical manuscript
```

## Output Format

The algorithm outputs standard PLY files with:
- **Format**: Binary little-endian
- **Precision**: Double precision coordinates (x, y, z)
- **Compatibility**: Compatible with major 3D processing software (Meshlab, CloudCompare, etc.)

Example output header:
```
ply
format binary_little_endian 1.0
comment Created by Open3D
element vertex 674018
property double x
property double y
property double z
end_header
```

## Performance

### Processing Statistics

The algorithm provides detailed processing feedback:
- Original point count
- Optimized k parameter value
- Processing time for parameter optimization
- Denoising execution time
- Final point count and removal statistics

### Example Output

```
Processing China dragon.ply: Original points = 892340
Optimized k=23 | iqr_ratio=2.0 | Time: 15.2s
Denoise time: 8.7s
Denoised points = 674018 (Removed 218322)
Saved to output/denoised_China dragon.ply
```

## Technical Implementation

### Key Classes

- **DensityBasedOutlierRemoval**: Core denoising engine
- **MetaLearner**: Parameter optimization system
- **GradientDescentOptimization**: Inner loop optimizer

### Memory Management

- Uses Open3D for efficient point cloud operations
- Implements parallel processing for k-nearest neighbors computation
- Supports streaming processing for very large datasets

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `k_neighbors` parameter or process smaller chunks
2. **Poor Results**: Adjust `iqr_ratio` parameter (try values between 1.0-3.0)
3. **Slow Processing**: The algorithm is computationally intensive; expect 10-30 seconds per 100k points

### Parameter Guidelines

- **k_neighbors**: Typically 10-50, automatically optimized
- **iqr_ratio**: Start with 1.5-2.0, increase for noisy data
- **voxel_size**: Default 0.05 works for most industrial/scanned data

## License

This software is provided as-is for research and development purposes.

## Citation

If you use this algorithm in your research, please cite the accompanying technical manuscript.

