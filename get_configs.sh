# enumerate cases
for len in 8 16 24 84 128 256 512 1024;
do
  CUDA_VISIBLE_DEVICES=3 python chatglm-test.py --seq-len $len --test-case 0 --engine-use
done