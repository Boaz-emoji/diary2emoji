python train.py --batch_size 32 \
                --device 0 \
                --max_length 64 \
                --model_path bert-base-uncased \
                --tokenizer_path bert-base-uncased \
                --data_path ./text_emoji.csv \
                --max_epochs 20 \
                --wandb_project EmojiDPR \

