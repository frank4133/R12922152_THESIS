import os
import sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.system('python test.py \
--dataset_name \
sice \
--under_exposure \
1 \
--over_exposure \
7 \
--evidence \
0 \
--checkpoint_path \
./checkpoints/TSMEF/08-17-13-13/300 \
--gpu_ids \
3 \
--mode \
Test \
--warp \
0 \
--network \
TSMEF \
--evidence_normalization \
linear \
'
)
