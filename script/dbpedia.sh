python src/train.py --task_name dbpedia \
                    --labeled_per_class 2500 \
                    --train_batch_size 32 \
                    --test_batch_size 128 \
                    --eval_step 200 \
                    --lr 5e-5 \
                    --epochs 10 \
                    --architecture pi \
                    --max_epochs 4 \
                    --last_epochs 2 \
                    --max_val 100 \
                    --aug
