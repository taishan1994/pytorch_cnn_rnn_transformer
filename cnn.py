import math
import logging
import numpy as np
from torch import nn
from typing import List
import os
import json

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report

from utils import set_seed, set_logger
set_seed(123)
logger = logging.getLogger(__name__)
set_logger(os.path.join('./logs/cnn.log'))

class CNN(nn.Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings"""

    def __init__(self, num_labels: int, vocab_size: int, in_word_embedding_dimension: int, out_channels: int = 256, kernel_sizes: List[int] = [1, 3, 5], stride_sizes: List[int] = None):
        nn.Module.__init__(self)
        self.config_keys = ['num_labels', 'vocab_size', 'in_word_embedding_dimension', 'out_channels', 'kernel_sizes']
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, in_word_embedding_dimension)
        self.embeddings_dimension = out_channels*len(kernel_sizes)
        self.convs = nn.ModuleList()
        self.fc = nn.Linear(self.embeddings_dimension, num_labels)

        in_channels = in_word_embedding_dimension
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int((kernel_size - 1) / 2)
            # (32 - 1 + (1-1)/2*2)/1 + 1 = 32
            # (32 - 3 + (3-1)/2*2)/1 + 1 = 32
            # (32 - 5 + (5-1)/2*2)/1 + 1 = 32
            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding_size)
            self.convs.append(conv)

    def forward(self, src):
        token_embeddings = self.embedding(src)
        token_embeddings = token_embeddings.transpose(1, -1) # [batchsize, in_word_embedding_dimension, max_seq_len]

        vectors = [conv(token_embeddings) for conv in self.convs]
        # for vec in vectors:
        #     print(vec.shape)
        out = torch.cat(vectors, 1)
        out = F.adaptive_max_pool1d(out, output_size=1).squeeze(-1)
        out = out.transpose(1, -1).contiguous() # [batchsize, out_channel*3, max_seq_len]
        out = self.fc(out)
        return out

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'cnn_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'cnn_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = CNN(**config)
        model.load_state_dict(weights)
        return model

class Args:
    data_dir = './data/'
    max_seq_len = 32
    batch_size = 64
    eval = True
    eval_step = 1000
    lr = 2e-4
    epoch = 8
    with open(data_dir + 'vocab.txt') as fp:
        vocab = fp.read().strip().split('\n')
    char2id = {}
    id2char = {}
    for i,char in enumerate(vocab):
        id2char[i] = char
        char2id[char] = i
    output_path = './checkpoints/cnn/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = output_path

    vocab_size = len(vocab)
    in_word_embedding_dimension = 300
    out_channels = 128
    kernel_sizes = [1, 3, 5]
    stride_sizes = None
    num_labels = 10
    dropout = 0.5

class ClassificationDataset(Dataset):
    def __init__(self, inputs, args):
        self.nums = len(inputs)
        self.args =args
        self.inputs = inputs

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = self.inputs[item]
        text = data[0]
        text = [i for i in text]
        if len(text) > self.args.max_seq_len:
            text = text[:self.args.max_seq_len]
        else:
            text = text + ['[PAD]'] * (self.args.max_seq_len - len(text))
        text_ids = torch.tensor([self.args.char2id.get(i, 1) for i in text]).long()
        label_ids = torch.tensor(int(data[1])).long()
        return text_ids, label_ids


class Classification:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = self.build_optimizer()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def data_loader(self, file_path):
        with open(file_path,'r') as fp:
            data = fp.read().strip().split('\n')
            data = [(d.split('\t')[0], d.split('\t')[1]) for d in data]
        classificationDataset = ClassificationDataset(data, self.args)
        data_loader = DataLoader(classificationDataset,batch_size=64, shuffle=True, num_workers=2)
        return data_loader

    def save_model(self, output_path: str):
        with open(os.path.join(output_path, 'lstm_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def train(self, train_loader, dev_loader=None):
        self.model.to(self.device)
        global_step = 0
        best_f1 = 0.0
        total_step = self.args.epoch * len(train_loader)
        for epoch in range(self.args.epoch):
            for step, data in enumerate(train_loader):
                self.model.train()
                text_ids, label_ids = data
                text_ids = text_ids.to(self.device) # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids)
                loss = self.criterion(output, label_ids)
                self.model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()
                logger.info('[train] epoch:{} step:{}/{} loss:{:.6f}'.format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if self.args.eval and global_step % args.eval_step == 0:
                    dev_loss, accuracy, precision, recall, f1 = self.dev(dev_loader)
                    logger.info('[dev] loss:{}, accuracy:{}, precision:{}, recall:{}, f1:{}'.format(
                       dev_loss, accuracy, precision, recall, f1
                    ))
                    if f1 > best_f1:
                        best_f1 = f1
                        self.model.save(self.args.output_path)

    def dev(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        pred_labels = []
        true_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                text_ids, label_ids = data
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids)
                loss = self.criterion(output, label_ids)
                total_loss += loss.item()
                preds = np.argmax(output.cpu().detach().numpy(), axis=1).tolist()
                trues = label_ids.cpu().detach().numpy().tolist()
                pred_labels.extend(preds)
                true_labels.extend(trues)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='micro')
        recall = recall_score(true_labels, pred_labels, average='micro')
        f1 = f1_score(true_labels, pred_labels, average='micro')
        return total_loss, accuracy, precision, recall, f1

    def test(self, dataloader, input_path):
        self.model.load(input_path)
        self.model.eval()
        self.model.to(self.device)
        pred_labels = []
        true_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for step, data in enumerate(dataloader):
                text_ids, label_ids = data
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids)
                loss = self.criterion(output, label_ids)
                total_loss += loss.item()
                preds = np.argmax(output.cpu().detach().numpy(), axis=1).tolist()
                trues = label_ids.cpu().detach().numpy().tolist()
                pred_labels.extend(preds)
                true_labels.extend(trues)
        return classification_report(true_labels, pred_labels)

    def predict(self, model, text):
        model.eval()
        model.to(self.device)
        text = [i for i in text]
        if len(text) > self.args.max_seq_len:
            text = text[:self.args.max_seq_len]
        else:
            text = text + ['[PAD]'] * (self.args.max_seq_len - len(text))
        text_ids = torch.tensor([self.args.char2id.get(i, 1) for i in text]).long()
        text_ids = text_ids.unsqueeze(0)
        text_ids = text_ids.to(self.device)
        with torch.no_grad():
            output = model(text_ids)
            output = np.argmax(output.cpu().detach().numpy(), axis=-1).tolist()
            output = output[0]
        return output

if __name__ == '__main__':
    args = Args()
    model = CNN(args.num_labels, args.vocab_size, args.in_word_embedding_dimension,
                args.out_channels, args.kernel_sizes, args.stride_sizes)
    classification = Classification(model, args)
    train_file = args.data_dir + 'train.txt'
    dev_file = args.data_dir + 'dev.txt'
    test_file = args.data_dir + 'test.txt'
    train_loader =  classification.data_loader(train_file)
    dev_loader = classification.data_loader(dev_file)
    test_loader = classification.data_loader(test_file)
    # classification.train(train_loader, dev_loader)
    # logger.info(classification.test(test_loader, args.input_path))
    model = model.load(args.input_path)
    with open('./data/test_my.txt','r') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            line = line.split('\t')
            text, label = line[0], line[1]
            logger.info('======================================')
            logger.info(text)
            pred = classification.predict(model, text)
            logger.info('真实标签：', label)
            logger.info('预测标签：', pred)