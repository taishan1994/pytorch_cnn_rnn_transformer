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
set_logger(os.path.join('./logs/lstm.log'))

class LSTM(nn.Module):
    """
    Bidirectional LSTM running over word embeddings.
    """

    def __init__(self, num_labels: int, vocab_size: int, word_embedding_dimension: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0,
                 bidirectional: bool = True):
        nn.Module.__init__(self)
        self.config_keys = ['num_labels','vocab_size','word_embedding_dimension', 'hidden_dim', 'num_layers', 'dropout', 'bidirectional']
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(word_embedding_dimension, hidden_dim, num_layers=num_layers, dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(self.embeddings_dimension, num_labels)

    def forward(self, src, sentence_lengths):
        # src:[batchsize, max_seq_len]
        token_embeddings = self.embedding(src)

        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, sentence_lengths, batch_first=True,
                                                   enforce_sorted=False)
        packed = self.encoder(packed)
        # 这里有一个参数可以控制填充到max_seq_len，total_length
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0] # [batchsize, 当前批次中句子的最大长度, hidden_dim*2]
        output = unpack.permute(0,2,1).contiguous()
        output = F.adaptive_max_pool1d(output, output_size=1).squeeze()
        output = self.fc(output)
        return output

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'lstm_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'lstm_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = LSTM(**config)
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
    output_path = './checkpoints/lstm/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = output_path

    vocab_size = len(vocab)
    word_embedding_dimension = 300
    hidden_dim = 128
    bidirectional = True
    num_layers = 1
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
                sequence_lengths = text_ids > 0
                sequence_lengths = torch.sum(sequence_lengths.long(), dim=1)
                # print(sequence_lengths)
                text_ids = text_ids.to(self.device) # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids, sequence_lengths) # [batchsize, max_seq_lem,, 128]
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
                sequence_lengths = text_ids > 0
                sequence_lengths = torch.sum(sequence_lengths.long(), dim=1)
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids, sequence_lengths)  # [max_seq_lem, batchsize, 128]
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
                sequence_lengths = text_ids > 0 # sequence_lengths是一个cpu上的张量
                sequence_lengths = torch.sum(sequence_lengths.long(), dim=1)
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids, sequence_lengths)  # [max_seq_lem, batchsize, 128]
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
        sequence_lengths = text_ids > 0
        sequence_lengths = torch.sum(sequence_lengths.long(), dim=1)
        text_ids = text_ids.to(self.device)
        with torch.no_grad():
            output = model(text_ids, sequence_lengths)
            output = np.argmax(output.cpu().detach().numpy(), axis=0).tolist()
        return output

if __name__ == '__main__':
    args = Args()
    model = LSTM(args.num_labels, args.vocab_size, args.word_embedding_dimension, args.hidden_dim,
                 args.num_layers, args.dropout, args.bidirectional)
    # for name,param in model.named_parameters():
    #     print(name)
    classification = Classification(model, args)
    train_file = args.data_dir + 'train.txt'
    dev_file = args.data_dir + 'dev.txt'
    test_file = args.data_dir + 'test.txt'
    train_loader =  classification.data_loader(train_file)
    dev_loader = classification.data_loader(dev_file)
    test_loader = classification.data_loader(test_file)
    classification.train(train_loader, dev_loader)
    logger.info(classification.test(test_loader, args.input_path))
    model = model.load(args.input_path)
    with open('./data/test_my.txt','r') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            line = line.split('\t')
            text, label = line[0], line[1]
            logger.info('======================================')
            logger.info(text)
            pred = classification.predict(model, text)
            logger.info('真实标签：' + str(label))
            logger.info('预测标签：' + str(pred))