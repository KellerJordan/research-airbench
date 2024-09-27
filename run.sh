#pip install airbench
#p=sweep_casted.py
p=svd_airbench.py
CUDA_VISIBLE_DEVICES=0 python $p
CUDA_VISIBLE_DEVICES=1 python $p
CUDA_VISIBLE_DEVICES=2 python $p
CUDA_VISIBLE_DEVICES=3 python $p
CUDA_VISIBLE_DEVICES=4 python $p
CUDA_VISIBLE_DEVICES=5 python $p
CUDA_VISIBLE_DEVICES=6 python $p
CUDA_VISIBLE_DEVICES=7 python $p

