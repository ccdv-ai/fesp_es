
python main.py \
    --train_config "config/train_config/train_config_pretrain_all.json" \
    --model_config "config/model_config/fesp/fesp_56_8_balanced.json"

python main.py \
    --train_config "config/train_config/train_config_pretrain_all.json" \
    --model_config "config/model_config/equalsurplus/equalsurplus_56_8_balanced.json"

python main.py \
    --train_config "config/train_config/train_config_pretrain_all.json" \
    --model_config "config/model_config/deeplift/deeplift.json"
