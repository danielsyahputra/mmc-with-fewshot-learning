CUDA_VISIBLE_DEVICES=$1 python src/train.py train.label_mode=make
CUDA_VISIBLE_DEVICES=$1 python src/train.py train.label_mode=model 
CUDA_VISIBLE_DEVICES=$1 python src/train.py train.label_mode=color 
CUDA_VISIBLE_DEVICES=$1 python src/train.py train.label_mode=make_model 
CUDA_VISIBLE_DEVICES=$1 python src/train.py train.label_mode=make_model_color 
