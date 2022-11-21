python src/train.py --task_name svhn \
                    --labeled_per_class 1000 \
                    --train_batch_size 100 \
                    --test_batch_size 256 \
                    --eval_step 200 \
                    --lr 1e-3 \
                    --epochs 300 \
                    --architecture temporal \
                    --max_val 30 \
                    --aug
