# ASUS-MEF
Multi-exposure Fusion for Dynamic Scenes
# Usage

### Environment
Python 3.11
```shell
# create conda environment
conda env create -f environment.yml

# the default env name is ASUS-MEF
conda activate ASUS-MEF
```

### Training
You can change the config in [start_train.py](./start_train.py)
```
python start_train.py
```
### Testing
To avoid OOM errors and dimension mismatches during concatenation, we resize the input to a multiple of 512 × 384, and then resize the output back.
The default input size is 3072 x 2048, which is controlled by the variable `scale`.
 - [Checkpoint](https://drive.google.com/drive/folders/1sKH9u8KoFyrwmffNqiVr4oRw5-coCuB3?usp=drive_link)
```
python start_test.py
```

### Citation
The codes are derived from the below research.
```
@article{liu2023holoco,
  title={HoLoCo: Holistic and local contrastive learning network for multi-exposure image fusion},
  author={Liu, Jinyuan and Wu, Guanyao and Luan, Junsheng and Jiang, Zhiying and Liu, Risheng and Fan, Xin},
  journal={Information Fusion},
  year={2023},
  publisher={Elsevier}
}
```