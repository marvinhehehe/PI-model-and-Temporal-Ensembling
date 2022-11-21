# Temporal-Ensembling-for-Semi-Supervised-Learning

This is the reimplement version of [Temporal-Ensembling-for-Semi-Supervised-Learning](https://arxiv.org/pdf/1610.02242.pdf) with Pytorch.

We reimplement $\Pi$-model and Temporal ensembling for both CV and NLP tasks, including CIFAR-10, SVHN, AG News, IMDB, DBpedia and Yahoo! Answers.

## Environment Configuration


- transformers==4.24.0

- spicy==0.16.0

- torch==1.12.1

- torchvision==0.13.1

- datasets==2.7.0


Run command below to install all the environment in need(**using python3**)

```shell
pip install -r requirements.txt
```

## Usage

```shell
python src/train.py --task_name ${TASK_NAME}$ \
                    --labeled_per_class ${LABELED_PER_CLASS}$ \
                    --train_batch_size ${TRAIN_BATCH_SIZE}$ \
                    --test_batch_size ${TEST_BATCH_SIZE}$ \
                    --eval_step ${EVAL_STEP}$ \
                    --lr ${LEARNING_RATE}$ \
                    --epochs ${EPOCHS}$ \
                    --architecture ${ARCHITECTURE}$ \
                    --max_epochs ${RAMP_UP_EPOCH}$ \
                    --last_epochs ${RAMP_DOWN_EPOCH}$ \
                    --max_val ${MAX_VAL}$ \
                    --aug
```

All the example scripts can be found in `src/script`

## Tips

Our implementation is not exactly the same as the official version. The original implementation based on Theano can be seen [here](https://github.com/smlaine2/tempens)