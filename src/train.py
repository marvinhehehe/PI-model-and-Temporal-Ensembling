import os.path
from argparse import ArgumentParser
import random
import numpy as np
import torch
from torch import nn
from transformers.utils.logging import set_verbosity_error
from utils import get_logger, weight_schedule, save_checkpoints, load_checkpoints, cutoff, img_aug, get_lr_lambda
from data import NLPDataModule, CVDataModule
from model import SequenceClassifier, CV_model
from evaluate import evaluate

log = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task_name", type=str, help="Name of the task.")
    parser.add_argument("--labeled_per_class", type=int,
                        help="How many labeled data per class in the dataset. In SVHN, this parameter means total amount of labeled data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Test batch size.")
    parser.add_argument("--eval_step", type=int, default=200, help="Evaluation the model every certain step.")
    parser.add_argument("--encoder_name_or_path", type=str, default="bert-base-uncased",
                        help="Name of the encoder in NLP task.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Total epoch.")
    parser.add_argument("--max_val", type=int, default=30,
                        help="Usually 100 in pi model and 30 in temporal ensembling.")
    parser.add_argument("--architecture", type=str, default="pi", help="The architecture of the training process.")
    parser.add_argument("--alpha", type=float, default=0.6, help="hyperparameter in rampup.")
    parser.add_argument("--max_epochs", type=int, default=80, help="hyperparameter in rampup.")
    parser.add_argument("--last_epochs", type=int, default=50, help="hyperparameter in rampup.")
    parser.add_argument("--aug", action='store_true', help="Do data augmentation or not.")
    parser.add_argument("--cutoff_rate", type=float, default=0.2, help="Cut off rate in data augmentation.")
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_batch(batch, args):
    if args.aug:
        if args.task_name in ["imdb", "ag_news", "yahoo", "dbpedia"]:
            return cutoff(batch, rate=args.cutoff_rate)
        else:
            return img_aug(batch, args.task_name)
    else:
        return batch


def train(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        log.info(f"Use gpu")
        device = torch.device("cuda")

    if args.task_name in ["imdb", "ag_news", "yahoo", "dbpedia"]:
        data_module = NLPDataModule(args)
        train_dataloader = data_module.train_dataloader() if args.architecture != "normal" else data_module.all_labeled_train_dataloader()
        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()
        num_labels = data_module.train_dataset.num_labels
        model = SequenceClassifier(args.encoder_name_or_path, num_labels).to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    else:
        data_module = CVDataModule(args)
        train_dataloader = data_module.train_dataloader() if args.architecture != "normal" else data_module.all_labeled_train_dataloader()
        val_dataloader = data_module.test_dataloader()
        test_dataloader = data_module.test_dataloader()
        num_labels = data_module.train_dataset.num_labels
        model = CV_model(num_labels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    step_per_epoch = np.floor(len(data_module.train_dataset) // args.train_batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=get_lr_lambda(args.epochs,
                                                                          step_per_epoch,
                                                                          args.max_epochs,
                                                                          args.last_epochs))

    root_dir = os.path.join("checkpoint", f"{args.task_name}_{args.labeled_per_class}_{args.architecture}")
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    log_file = open(os.path.join(root_dir, "log.txt"), "w")
    current_step = 0
    best_eval_loss = 10000.
    best_acc = 0.
    Z = torch.zeros(len(data_module.train_dataset), num_labels).to(device)
    z = torch.zeros(len(data_module.train_dataset), num_labels).to(device)
    total_loss = 0.
    total_labeled = args.labeled_per_class if args.task_name == "svhn" else args.labeled_per_class * num_labels
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss = nn.MSELoss()

    log.info("Start Training")
    log.info(f"Total Training Data: {len(train_dataloader.dataset)}")
    log.info(f"Labeled Data: {total_labeled}")
    for epoch in range(args.epochs):
        log.info(f"epoch {epoch}:")
        w = weight_schedule(epoch, args.max_epochs, args.max_val, total_labeled,
                            len(data_module.train_dataset))
        for batch in train_dataloader:
            model.train()
            loss = 0.
            batch = batch.to(device)
            if args.architecture == "pi":
                logits_1 = model(get_batch(batch, args))
                with torch.no_grad():
                    logits_2 = model(get_batch(batch, args))
                if torch.sum(batch.labels, dim=0) != -args.train_batch_size:
                    loss = cross_entropy_loss(logits_1, batch.labels)
                loss += w * mse_loss(logits_1, logits_2.detach())
            elif args.architecture == "temporal":
                logits = model(get_batch(batch, args))
                single_ids = batch.ids[:, 0]
                if torch.sum(batch.labels, dim=0) != -args.train_batch_size:
                    loss = cross_entropy_loss(logits, batch.labels)
                loss += w * mse_loss(logits, z[single_ids].detach())
                Z.scatter_(0, batch.ids, args.alpha * Z[single_ids] + (1. - args.alpha) * logits)
            elif args.architecture == "normal":
                logits = model(get_batch(batch, args))
                loss = cross_entropy_loss(logits, batch.labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            current_step += 1
            if current_step % 50 == 0:
                log.info(f"training loss: {(total_loss / current_step):5.4f}")
            if current_step % args.eval_step == 0:
                eval_loss, acc = evaluate(model,
                                          val_dataloader,
                                          device,
                                          log_file)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    checkpoints = {
                        "step": current_step,
                        "model": model.state_dict()
                    }
                    save_checkpoints(os.path.join(root_dir, "best_loss_checkpoint"), checkpoints)
                if acc > best_acc:
                    best_acc = acc
                    checkpoints = {
                        "step": current_step,
                        "model": model.state_dict()
                    }
                    save_checkpoints(os.path.join(root_dir, "best_acc_checkpoint"), checkpoints)
        if args.architecture == "temporal":
            z = Z * (1. / (1. - args.alpha ** (epoch + 1)))
    ckpt = load_checkpoints(os.path.join(root_dir, "best_loss_checkpoint"), device)
    model.load_state_dict(ckpt["model"])
    log.info("Test the best loss checkpoints.")
    evaluate(model,
             test_dataloader,
             device,
             log_file)
    ckpt = load_checkpoints(os.path.join(root_dir, "best_acc_checkpoint"), device)
    model.load_state_dict(ckpt["model"])
    log.info("Test the best acc checkpoints.")
    evaluate(model,
             test_dataloader,
             device,
             log_file)
    log_file.close()


if __name__ == '__main__':
    set_verbosity_error()

    args = parse_args()
    log.info(f'args:{args}')
    log.info(f"Set seed to {args.seed}")
    seed_everything(args.seed)
    train(args)
