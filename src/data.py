import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os

transform_type = {
    "cifar-10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
    ]),
    "svhn": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])}


class NLPDataModule:
    def __init__(self, args):
        dataset_list = {"imdb": IMDBDataset,
                        "ag_news": AgnewsDataset,
                        "yahoo": YahooDataset,
                        "dbpedia": DBPEDIADataset}

        train_text, train_label, val_text, val_label, test_text, test_label, all_labeled_train_text, all_labeled_train_label = self.dataset_split(
            args.task_name,
            args.labeled_per_class)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        dataset_class = dataset_list[args.task_name]

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        self.train_dataset = dataset_class(train_text, train_label, tokenizer)
        self.val_dataset = dataset_class(val_text, val_label, tokenizer)
        self.test_dataset = dataset_class(test_text, test_label, tokenizer)
        self.all_labeled_train_dataset = dataset_class(all_labeled_train_text, all_labeled_train_label, tokenizer)

        self.collator = nlpcollator

    def dataset_split(self, task_name, labeled_per_class):
        train_idx = []
        val_idx = []
        all_labeled_train_idx = []
        train_label = []
        text_name = "content" if task_name == "dbpedia" else "text"
        if task_name == "imdb":
            if not os.path.isdir(os.path.join("data", "imdb")):
                os.makedirs(os.path.join("data", "imdb"))
            data = load_dataset("imdb", cache_dir="./data/imdb")
            for i in range(2):
                idxs = list(range(12500 * i, 12500 * (i + 1)))
                random.shuffle(idxs)
                train_idx += idxs[:5000 + labeled_per_class]
                val_idx += idxs[-2000:]
                all_labeled_train_idx += idxs[:labeled_per_class]
                tmp_label = data["train"][idxs[:5000 + labeled_per_class]]["label"]
                tmp_label = [label if index < labeled_per_class else -1 for index, label in enumerate(tmp_label)]
                train_label += tmp_label
        elif task_name == "ag_news":
            if not os.path.isdir(os.path.join("data", "ag_news")):
                os.makedirs(os.path.join("data", "ag_news"))
            data = load_dataset("ag_news", cache_dir="./data/ag_news")
            labels = np.array([item["label"] for item in data["train"]])
            for i in range(4):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_idx += list(idxs[:5000 + labeled_per_class])
                val_idx += list(idxs[-2000:])
                all_labeled_train_idx += list(idxs[:labeled_per_class])
                tmp_label = data["train"][idxs[:5000 + labeled_per_class]]["label"]
                tmp_label = [label if index < labeled_per_class else -1 for index, label in enumerate(tmp_label)]
                train_label += tmp_label
        elif task_name == "yahoo":
            if not os.path.isdir(os.path.join("data", "yahoo")):
                os.makedirs(os.path.join("data", "yahoo"))
            data = load_dataset("Brendan/yahoo_answers", cache_dir="./data/yahoo")
            labels = np.array([item["label"] for item in data["train"]])
            for i in range(10):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_idx += list(idxs[:5000 + labeled_per_class])
                val_idx += list(idxs[-5000:])
                all_labeled_train_idx += list(idxs[:labeled_per_class])
                tmp_label = data["train"][idxs[:5000 + labeled_per_class]]["label"]
                tmp_label = [label if index < labeled_per_class else -1 for index, label in enumerate(tmp_label)]
                train_label += tmp_label
        elif task_name == "dbpedia":
            if not os.path.isdir(os.path.join("data", "dbpedia")):
                os.makedirs(os.path.join("data", "dbpedia"))
            data = load_dataset("dbpedia_14", cache_dir="./data/dbpedia")
            for i in range(14):
                idxs = list(range(40000 * i, 40000 * (i + 1)))
                random.shuffle(idxs)
                train_idx += idxs[:5000 + labeled_per_class]
                val_idx += idxs[-2000:]
                all_labeled_train_idx += idxs[:labeled_per_class]
                tmp_label = data["train"][idxs[:5000 + labeled_per_class]]["label"]
                tmp_label = [label if index < labeled_per_class else -1 for index, label in enumerate(tmp_label)]
                train_label += tmp_label
        return data["train"][train_idx][text_name], train_label, data["train"][val_idx][text_name], \
               data["train"][val_idx]["label"], data["test"][text_name], data["test"]["label"], \
               data["train"][all_labeled_train_idx][text_name], data["train"][all_labeled_train_idx]["label"]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def all_labeled_train_dataloader(self):
        return DataLoader(
            dataset=self.all_labeled_train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


class CVDataModule:
    def __init__(self, args):
        dataset_list = {"cifar-10": CIFAR10Dataset,
                        "svhn": SVHNDataset}

        train_set, test_set, all_labeled_set = self.dataset_split(
            args.task_name,
            args.labeled_per_class,
            args.aug)

        dataset_class = dataset_list[args.task_name]

        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        self.train_dataset = dataset_class(train_set)
        self.test_dataset = dataset_class(test_set)
        self.all_labeled_train_dataset = dataset_class(all_labeled_set)

        self.collator = cvcollator

    def dataset_split(self, task_name, labeled_per_class, aug=False):
        new_trainset = []
        labeled_trainset = []
        if task_name == "cifar-10":
            if not os.path.isdir(os.path.join("data", "cifar-10")):
                os.makedirs(os.path.join("data", "cifar-10"))
            trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True,
                                                    transform=transforms.ToTensor())
            testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True,
                                                   transform=transform_type["cifar-10"])
            labels = np.array([item[1] for item in trainset])
            for i in range(10):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_idx = idxs[:labeled_per_class]
                unlabeled_idx = idxs[labeled_per_class:]
                for idx in train_idx:
                    new_trainset.append(trainset[idx])
                    labeled_trainset.append(trainset[idx])
                for idx in unlabeled_idx:
                    new_trainset.append((trainset[idx][0], -1))
        elif task_name == "svhn":
            if not os.path.isdir(os.path.join("data", "svhn")):
                os.makedirs(os.path.join("data", "svhn"))
            trainset = torchvision.datasets.SVHN(root='./data/svhn', split="train", download=True,
                                                 transform=transforms.ToTensor())
            testset = torchvision.datasets.SVHN(root='./data/svhn', split="test", download=True,
                                                transform=transform_type["svhn"])
            if labeled_per_class > 45690:
                labeled_trainset, unlabeled_trainset = random_split(trainset,
                                                                    [labeled_per_class,
                                                                     len(trainset) - labeled_per_class])
                for item in labeled_trainset:
                    new_trainset.append(item)
                for item in unlabeled_trainset:
                    new_trainset.append((item[0], -1))
            else:
                labeled_per_class = labeled_per_class / 10
                labels = np.array([item[1] for item in trainset])
                for i in range(10):
                    idxs = np.where(labels == i)[0]
                    np.random.shuffle(idxs)
                    train_idx = idxs[:labeled_per_class]
                    unlabeled_idx = idxs[labeled_per_class:]
                    for idx in train_idx:
                        new_trainset.append(trainset[idx])
                        labeled_trainset.append(trainset[idx])
                    for idx in unlabeled_idx:
                        new_trainset.append((trainset[idx][0], -1))
        return new_trainset, testset, labeled_trainset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def all_labeled_train_dataloader(self):
        return DataLoader(
            dataset=self.all_labeled_train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


class IMDBDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_seq_len=256):
        tokenized_input = tokenizer(text,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_seq_len,
                                    return_tensors="pt")
        self.input_ids = tokenized_input.input_ids
        self.token_type_ids = tokenized_input.token_type_ids
        self.attention_mask = tokenized_input.attention_mask
        self.label = label
        self.num_labels = 2

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], \
               self.label[item]


class AgnewsDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_seq_len=256):
        tokenized_input = tokenizer(text,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_seq_len,
                                    return_tensors="pt")
        self.input_ids = tokenized_input.input_ids
        self.token_type_ids = tokenized_input.token_type_ids
        self.attention_mask = tokenized_input.attention_mask
        self.label = label
        self.num_labels = 4

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], \
               self.label[item]


class YahooDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_seq_len=256):
        tokenized_input = tokenizer(text,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_seq_len,
                                    return_tensors="pt")
        self.input_ids = tokenized_input.input_ids
        self.token_type_ids = tokenized_input.token_type_ids
        self.attention_mask = tokenized_input.attention_mask
        self.label = label
        self.num_labels = 10

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], \
               self.label[item]


class DBPEDIADataset(Dataset):
    def __init__(self, text, label, tokenizer, max_seq_len=256):
        tokenized_input = tokenizer(text,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_seq_len,
                                    return_tensors="pt")
        self.input_ids = tokenized_input.input_ids
        self.token_type_ids = tokenized_input.token_type_ids
        self.attention_mask = tokenized_input.attention_mask
        self.label = label
        self.num_labels = 14

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], \
               self.label[item]


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.num_labels = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.data[item][0], self.data[item][1]


class SVHNDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.num_labels = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [item] * self.num_labels, self.data[item][0], self.data[item][1]


class Batch:
    def __init__(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, imgs=None,
                 labels=None, ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.imgs = imgs
        self.labels = labels
        self.ids = ids

    def to(self, device):
        self.input_ids = self.input_ids.to(device) if self.input_ids is not None else None
        self.attention_mask = self.attention_mask.to(device) if self.attention_mask is not None else None
        self.token_type_ids = self.token_type_ids.to(device) if self.token_type_ids is not None else None
        self.position_ids = self.position_ids.to(device) if self.position_ids is not None else None
        self.imgs = self.imgs.to(device) if self.imgs is not None else None
        self.labels = self.labels.to(device) if self.labels is not None else None
        self.ids = self.ids.to(device) if self.ids is not None else None
        return self


def nlpcollator(items):
    ids, input_ids, token_type_ids, attention_mask, labels = zip(*items)
    input_ids = torch.stack(input_ids, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    ids = torch.tensor(ids)
    labels = torch.tensor(labels)
    position_ids = torch.arange(input_ids.size(-1), dtype=torch.long).expand((1, -1))
    return Batch(input_ids=input_ids,
                 token_type_ids=token_type_ids,
                 attention_mask=attention_mask,
                 ids=ids,
                 labels=labels,
                 position_ids=position_ids)


def cvcollator(items):
    ids, imgs, labels = zip(*items)
    imgs = torch.stack(imgs, dim=0)
    ids = torch.tensor(ids)
    labels = torch.tensor(labels)
    return Batch(imgs=imgs,
                 ids=ids,
                 labels=labels)
