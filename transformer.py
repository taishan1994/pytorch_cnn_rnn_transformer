import math
import json
import os
import logging
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch import Tensor, optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from utils import set_seed, set_logger

set_seed(123)
logger = logging.getLogger(__name__)
set_logger(os.path.join('./logs/transformer.log'))


# self.embedding = nn.Embedding(vocab_size,d_model)
# self.embedding.weight = nn.Embedding.from_pretrained(embedding_weight)
# self.embedding.weight.requires_grad = False if fix_embedding else True # 设定emb是否随训练更新


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len,  embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, num_label: int, pe_dropout: float, ec_dropout: float):
        nn.Module.__init__(self)
        self.config_keys = ['ntoken', 'd_model', 'nhead', 'd_hid',
                            'nlayers', 'num_label', 'pe_dropout', 'ec_dropout']
        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.num_label = num_label
        self.pe_dropout = pe_dropout
        self.ec_dropout = ec_dropout

        self.pos_encoder = PositionalEncoding(d_model, pe_dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, ec_dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, encoder_norm)
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=0)

        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, num_label)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_key_padding_mask : Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_labels]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.fc1(output)  # [batchsize, max_seq_len, 128]
        output = output.permute(0, 2, 1).contiguous()
        output = F.adaptive_avg_pool1d(output, output_size=1).squeeze()
        output = self.fc2(output)
        return output

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'transformer_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'transformer_config.json'), 'r') as fIn:
            config = json.load(fIn)
        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = TransformerModel(**config)
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
    for i, char in enumerate(vocab):
        id2char[i] = char
        char2id[char] = i
    output_path = './checkpoints/transformer/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    input_path = output_path

    ntoken = len(vocab)
    d_model = 256
    nhead = 4
    d_hid = 512
    nlayers = 1
    num_label = 10
    pe_dropout = 0.5
    ec_dropout = 0.5


class ClassificationDataset(Dataset):
    def __init__(self, inputs, args):
        self.nums = len(inputs)
        self.args = args
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
        with open(file_path, 'r') as fp:
            data = fp.read().strip().split('\n')
            data = [(d.split('\t')[0], d.split('\t')[1]) for d in data]
        classificationDataset = ClassificationDataset(data, self.args)
        data_loader = DataLoader(classificationDataset, batch_size=64, shuffle=True, num_workers=2)
        return data_loader

    def build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def train(self, train_loader, dev_loader=None):
        self.model.train()
        self.model.to(self.device)
        global_step = 0
        best_f1 = 0.0
        total_step = self.args.epoch * len(train_loader)
        for epoch in range(self.args.epoch):
            for step, data in enumerate(train_loader):
                text_ids, label_ids = data
                attn_masks = (text_ids == 0)
                attn_masks = attn_masks.to(self.device)
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                label_ids = label_ids.to(self.device)
                output = self.model(text_ids, attn_masks)  # [max_seq_lem, batchsize, 128]
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
                attn_masks = (text_ids == 0)
                attn_masks = attn_masks.to(self.device)
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                output = self.model(text_ids, attn_masks)
                label_ids = label_ids.to(self.device)
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
                attn_masks = (text_ids == 0)
                attn_masks = attn_masks.to(self.device)
                text_ids = text_ids.to(self.device)  # [batchsize, max_seq_len]
                output = self.model(text_ids, attn_masks)
                label_ids = label_ids.to(self.device)
                loss = self.criterion(output, label_ids)
                total_loss += loss.item()
                preds = np.argmax(output.cpu().detach().numpy(), axis=1).tolist()
                trues = label_ids.cpu().detach().numpy().tolist()
                pred_labels.extend(preds)
                true_labels.extend(trues)
        # print(true_labels)
        # print(pred_labels)
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
        attn_masks = (text_ids == 0)
        attn_masks = attn_masks.to(self.device)
        text_ids = text_ids.to(self.device)
        with torch.no_grad():
            output = model(text_ids, attn_masks)
            output = np.argmax(output.cpu().detach().numpy(), axis=0).tolist()
        return output


if __name__ == '__main__':
    args = Args()
    model = TransformerModel(args.ntoken, args.d_model, args.nhead, args.d_hid,
                             args.nlayers, args.num_label, args.pe_dropout, args.ec_dropout)
    # for name,param in model.named_parameters():
    #     print(name)
    classification = Classification(model, args)
    train_file = args.data_dir + 'train.txt'
    dev_file = args.data_dir + 'dev.txt'
    test_file = args.data_dir + 'test.txt'
    train_loader = classification.data_loader(train_file)
    dev_loader = classification.data_loader(dev_file)
    test_loader = classification.data_loader(test_file)
    # classification.train(train_loader, dev_loader)
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