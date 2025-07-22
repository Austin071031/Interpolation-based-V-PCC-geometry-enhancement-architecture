# Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach

This repository contains the official implementation for the paper:  
**"Video-based Point Cloud Compression Quality Enhancement Using Deep Learning Approach"**

## Prerequisites

Before you begin, ensure you have the following installed:
- Python: 3.8.20
- PyTorch: 1.9.1+cu111 (Compiled with CUDA 11.1 support)
- CUDA: 11.1 (Version used by PyTorch)
- MinkowskiEngine: 0.5.4
- Block Dataset: [Download sample dataset](https://onedrive.com) (replace with actual link)

## How to Generate the Block Dataset

### For BaseNet
Prepare paired blocks of point clouds for training:

1. **Prepare Initial Data**  
   Organize two folders:
   - `origin_folder`: Original uncompressed point cloud files (Ground Truth)
   - `compress_folder`: Corresponding VPCC-compressed point cloud files

2. **Cut Point Clouds into Blocks**  
   ```bash
   python src/cut_block.py \
   --compress_folder=/path/to/compress_folder \
   --origin_folder=/path/to/origin_folder \
   --block_output_folder=/path/to/block_output
