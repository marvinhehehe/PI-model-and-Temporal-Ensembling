from transformers import BertModel
from torch import nn
from torch.nn.utils import weight_norm


class SequenceClassifier(nn.Module):
    def __init__(self, encoder_name_or_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(encoder_name_or_path)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, batch):
        bert_output = self.bert(input_ids=batch.input_ids,
                                token_type_ids=batch.token_type_ids,
                                attention_mask=batch.attention_mask,
                                position_ids=batch.position_ids)
        pooled_output = self.dropout(bert_output.pooler_output)
        logits = self.classifier(pooled_output)
        return logits


class gaussian_noise(nn.Module):
    def __init__(self, mean=0, std=0.15):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        noise = input.data.new(input.size()).normal_(self.mean, self.std)
        return input + noise


class CV_model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.gn = gaussian_noise()
        self.conv1a = weight_norm(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding='same'))
        self.conv1b = weight_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'))
        self.conv2a = weight_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'))
        self.conv2b = weight_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'))

        self.conv3a = weight_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='valid'))
        self.conv3b = weight_norm(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1))
        self.conv3c = weight_norm(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, batch):
        x = batch.imgs
        x = self.gn(x)  # gaussian noise
        x = self.lrelu(self.conv1a(x))  # conv1a
        x = self.lrelu(self.conv1b(x))  # conv1b
        x = self.lrelu(self.conv1b(x))  # conv1c
        x = self.maxpool(x)  # pool1
        x = self.dropout(x)  # drop1
        x = self.lrelu(self.conv2a(x))  # conv2a
        x = self.lrelu(self.conv2b(x))  # conv2b
        x = self.lrelu(self.conv2b(x))  # conv2c
        x = self.maxpool(x)  # pool2
        x = self.dropout(x)  # drop2
        x = self.lrelu(self.conv3a(x))  # conv3a
        x = self.lrelu(self.conv3b(x))  # conv3b
        x = self.lrelu(self.conv3c(x))  # conv3c
        x = self.gap(x)  # pool3
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)  # dense
        return x
