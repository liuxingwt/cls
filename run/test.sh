export CUDA_VISIBLE_DEVICES=0
MOBILE_MEMORY=20211203

OUTPUT=./results/out_${MOBILE_MEMORY}.txt/
mkdir -p ${OUTPUT}


python3 test.py  \
--arch resnet10 \
--valid_db "./test.txt" \
--ckpt "./training-runs/checkpoint_20211129_2/checkpoint-100.pth.tar" \
--out ${OUTPUT}
