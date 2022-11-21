python src/train.py --task_name cifar-10 \
                    --labeled_per_class 400 \
                    --train_batch_size 100 \
                    --test_batch_size 256 \
                    --eval_step 200 \
                    --lr 3e-3 \
                    --epochs 300 \
                    --architecture pi \
                    --max_val 100 \
                    --aug
