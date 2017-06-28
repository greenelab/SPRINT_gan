CUDA_VISIBLE_DEVICES="0" python dp_gan.py --noise 2 --clip_value 0.0001 --epochs 500 --lr 2e-05 --batch_size 1 --prefix 2_0.0001_2e-05_ &
CUDA_VISIBLE_DEVICES="0" python dp_gan.py --noise 2 --clip_value 0.00001 --epochs 500 --lr 2e-05 --batch_size 1 --prefix 2_0.00001_2e-05_ &
CUDA_VISIBLE_DEVICES="0" python dp_gan.py --noise 2 --clip_value 0.0001 --epochs 500 --lr 0.0002 --batch_size 1 --prefix 2_0.0001_2e-05_ &
CUDA_VISIBLE_DEVICES="1" python dp_gan.py --noise 1 --clip_value 0.0001 --epochs 500 --lr 2e-05 --batch_size 1 --prefix 1_0.0001_2e-05_ &
CUDA_VISIBLE_DEVICES="1" python dp_gan.py --noise 1 --clip_value 0.00001 --epochs 500 --lr 2e-05 --batch_size 1 --prefix 1_0.00001_2e-05_ &
CUDA_VISIBLE_DEVICES="1" python dp_gan.py --noise 1 --clip_value 0.0001 --epochs 500 --lr 0.0002 --batch_size 1 --prefix 1_0.0001_2e-05_ &
CUDA_VISIBLE_DEVICES="1" python dp_gan.py --noise 1 --clip_value 0.00001 --epochs 500 --lr 0.0002 --batch_size 1 --prefix 1_0.00001_2e-05_ &
