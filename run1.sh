#CUDA_VISIBLE_DEVICES=3 python tools/train.py --alpha 0.1 --beta 10 --gamma 0.2 --dropout 0.5 --wd 0.000001
CUDA_VISIBLE_DEVICES=2 python tools/test.py > "coco_"$(date +%s)"_0.1_10_0.2_0.5_0.000001.txt" 2>&1
        
