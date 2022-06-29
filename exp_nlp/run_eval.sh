
python main.py \
    --model_config "config/model_config/fesp/fesp_1_1.json" \
    --train_config "config/train_config/train_config_pretrain.json"

python main.py \
    --model_config "config/model_config/equalsurplus/equalsurplus_1_1.json" \
    --train_config "config/train_config/train_config_pretrain.json"

python main.py \
    --model_config "config/model_config/occlusion/occlusion_1_1.json" \
    --train_config "config/train_config/train_config_pretrain.json"

python main.py \
    --model_config "config/model_config/shapexplainer/shapexplainer.json" \
    --train_config "config/train_config/train_config_pretrain.json"

python main.py \
    --model_config "config/model_config/deeplift/deeplift.json" \
    --train_config "config/train_config/train_config_pretrain.json"
