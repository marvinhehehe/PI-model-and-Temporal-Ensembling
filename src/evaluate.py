import torch
from torch import nn
import numpy as np
from utils import get_logger

log = get_logger(__name__)


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def evaluate(model, dataloader, device, log_file):
    model.eval()
    eval_loss = 0.0
    preds = []
    labels = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)

        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
        tmp_eval_loss = cross_entropy_loss(logits, batch.labels)

        labels.append(batch.labels)
        eval_loss += tmp_eval_loss.item()
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / len(dataloader)

    labels = torch.cat(labels, dim=0)
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    acc = simple_accuracy(preds, labels.cpu().numpy())

    log.info(f"Evaluate end | eval loss {eval_loss:5.4f} | acc {acc:5.4f}")
    log_file.write(f"{acc:5.4f}" + '\n')

    return eval_loss, acc
