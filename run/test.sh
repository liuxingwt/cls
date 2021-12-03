export CUDA_VISIBLE_DEVICES=0

OUTPUT=./results/
mkdir -p ${OUTPUT}

MOBILE_MEMORY=20211203
OUTPUTFILE=./${OUTPUT}/out_${MOBILE_MEMORY}.txt
touch ${OUTPUTFILE}


python3 test.py  \
--arch resnet10 \
--valid_db "./test.txt" \
--ckpt "./training-runs/checkpoint_20211129_2/checkpoint-100.pth.tar" \
--out ${OUTPUTFILE}
