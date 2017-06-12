CUDA_VISIBLE_DEVICES=0 python dp_gan.py --clip_value 0.0001 --noise 8 --epochs 500 --lr 0.002 --batch_size 100 --prefix p0_ --seed 123
CUDA_VISIBLE_DEVICES=0 python dp_gan.py --clip_value 0.0001 --noise 8 --epochs 500 --lr 0.002 --batch_size 100 --prefix p1_ --seed 234
CUDA_VISIBLE_DEVICES=0 python dp_gan.py --clip_value 0.0001 --noise 8 --epochs 500 --lr 0.002 --batch_size 100 --prefix p2_ --seed 567
CUDA_VISIBLE_DEVICES=0 python dp_gan.py --clip_value 0.0001 --noise 8 --epochs 500 --lr 0.002 --batch_size 100 --prefix p3_ --seed 678
CUDA_VISIBLE_DEVICES=0 python dp_gan.py --clip_value 0.0001 --noise 8 --epochs 500 --lr 0.002 --batch_size 100 --prefix p4_ --seed 789
