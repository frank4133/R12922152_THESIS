# NTU Master Thesis
Two-Phase Progressive Multiple Exposure Fusion via Intermediate Exposure Generation

### Environment
Python 3.11
```shell
# create conda environment
conda env create -f environment.yml

# the default env name is ASUS-MEF
conda activate holoco
```

### Training
You can change the config in [start_train.py](./start_train.py)
```
python start_train.py
```
### Testing
To avoid OOM errors and dimension mismatches during concatenation, we resize the input to around FHD, which is controlled by the variable `scale`.
 - [Checkpoint](https://drive.google.com/drive/folders/1sKH9u8KoFyrwmffNqiVr4oRw5-coCuB3?usp=drive_link)
```
python start_test.py
```

- [Output]()


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

@article{yan2021dual,
  title={Dual-attention-guided network for ghost-free high dynamic range imaging},
  author={Yan, Qingsen and Gong, Dong and Shi, Javen Qinfeng and van den Hengel, Anton and Shen, Chunhua and Reid, Ian and Zhang, Yanning},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2021},
  publisher={Springer}
}

@article{yan2019attention,
  title={Attention-guided Network for Ghost-free High Dynamic Range Imaging},
  author={Yan, Qingsen and Gong, Dong and Shi, Qinfeng and Hengel, Anton van den and Shen, Chunhua and Reid, Ian and Zhang, Yanning},
  journal={IEEE Conference on Compute rVision and Pattern Recognition (CVPR)},
  year={2019}
  pages={1751-1760}
}
```