# Interpolation-based V-PCC geometry enhancement architecture

This repository contains the key official implementation for the paper:  
**"Complement decoded point cloud with coordinate adjustment for video-based point cloud compression"**

## Prerequisites

Before you begin, ensure you have the following installed:
- Python: 3.7.12
- PyTorch: 1.7.0+cu110
- CUDA: 11.0
- MinkowskiEngine: 0.5.4

## Prepare the Block Dataset

1. **Prepare Initial Data**
     
   Organize two folders:
   - `origin_folder`: Original uncompressed point cloud files (Ground Truth)
   - `compress_folder`: Corresponding VPCC-compressed point cloud files

2. **Cut Point Clouds into Blocks**

   Configure the parameters in the file **pc_cube_split_overlap.py**
   ```bash
   # pc_file = "/point-cloud-file-path/"
   # save_cube_file = "/save-blocks-file-path/"
   # cube_size = define the size of block, default 100
   # stride = define overslap between blocks, default 95
   ```

   Run command in terminal
   ```bash
   python ply_processing/pc_cube_split_overlap.py

3. **Generate Interpolation Points**
    Configure the parameters in the file **gen_interpolation.py**
   ```bash
    # interpolation_dir_noisyinput = '/save-interpolation-points-patch/'
    # noisy, filename = prepare_data('/block-dataset-path/', 'dataset_folder')

    scale_factor = define the ratio of interpolation generated (default 0.5)
   ```

   Run command in terminal
   ```bash
   python ply_processing/gen_interpolation.py
   ```

## Training
```bash
python src/genpd_train.py \
--dataset_8i=NOISY_TRAINING_DATA
--dataset_8i_GT=GROUNDTRUTH_TRAINING_DATA
--dataset_8i_val=NOISY_TESTING_DATA
--dataset_8i_val_GT=GROUNDTRUTH_TESTING_DATA


```
   
   
