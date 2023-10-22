from transformers import AlbertTokenizer, AlbertModel, AlbertPreTrainedModel, AlbertConfig, AlbertForPreTraining, \
    AutoModelForMaskedLM
from transformers import BigBirdModel, BigBirdPreTrainedModel, BigBirdConfig
from transformers import LongformerModel, LongformerConfig, LongformerPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging
from torch.utils.data import DataLoader
from torch import logit, nn
from torch.nn import functional as F
from d2l import torch as d2l
import torch
import os

logging.set_verbosity_error()


class TextCNN(nn.Module):
    def __init__(self, filter_sizes, num_filter, num_labels):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filter_total = num_filter * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, num_labels, bias=False)
        self.bias = nn.Parameter(torch.ones([num_labels]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filter, kernel_size=(size, 768)) for size in filter_sizes
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))
            mp = nn.MaxPool2d(kernel_size=(12 - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, 3)  # [bs, h=1, w=1, channel=192]
        h_pool_flat = self.dropout(torch.reshape(h_pool, [-1, self.num_filter_total]))

        output = self.Weight(h_pool_flat) + self.bias  # [bs, n_class]

        return output


class ALBertAndTextCnnForSeq(AlbertPreTrainedModel):
    def __init__(self, config, filters, filter_num):
        super(ALBertAndTextCnnForSeq, self).__init__(config)
        self.out_channels = 32
        # 获得预训练模型的参数
        self.config = AlbertConfig(config)
        # 标签数量
        self.num_labels = 31
        # albert模型
        self.albert = AlbertModel(config)
        # self.albert = AutoModelForMaskedLM(config)

        self.relu = nn.ReLU()

        self.filter = filters
        self.filter_num = filter_num

        # 最后的分类层
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # textCnn层
        self.text_cnn = TextCNN(filter_sizes=filters, num_filter=filter_num,
                                num_labels=self.num_labels)

        # 初始化权重
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取模型结果
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # shape = (batch_size=16, sequence_length=512, hidden_size=768)
        # hidden_states = outputs[0]
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
        # cls_embeddings = self.relu(hidden_states)
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # logits会返回一个还未经过
        logits = self.text_cnn(cls_embeddings)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ALBertForSeq(AlbertPreTrainedModel):

    def __init__(self, config):
        super(ALBertForSeq, self).__init__(config)

        self.config = AlbertConfig(config)
        self.num_labels = 64
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        pooler_output = outputs[1]
        pooler_output = self.dropout(pooler_output)

        logits = self.classifier(pooler_output)
        loss = None

        if labels is not None:
            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class BigBirdForSeq(BigBirdPreTrainedModel):

    def __init__(self, config):
        super(BigBirdForSeq, self).__init__(config)

        self.config = BigBirdConfig(config)
        self.num_labels = 64
        self.model = BigBirdModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        pooler_output = outputs[1]
        pooler_output = self.dropout(pooler_output)

        logits = self.classifier(pooler_output)
        loss = None

        if labels is not None:
            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class LongformerForSeq(LongformerPreTrainedModel):

    def __init__(self, config):
        super(LongformerForSeq, self).__init__(config)

        self.config = LongformerConfig(config)
        self.num_labels = 64
        self.model = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        pooler_output = outputs[1]
        pooler_output = self.dropout(pooler_output)

        logits = self.classifier(pooler_output)
        loss = None

        if labels is not None:
            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
