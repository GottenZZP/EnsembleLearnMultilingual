from process import InputDataSet
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertPreTrainedModel, DistilBertConfig
from transformers import DebertaModel, DebertaTokenizer, DebertaConfig, DebertaPreTrainedModel
from transformers import DebertaV2Model, DebertaV2Config, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging
from torch import nn
from torch.nn import functional as F
import torch

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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))
            mp = nn.MaxPool2d(kernel_size=(7 - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, 3)  # [bs, h=1, w=1, channel=192]
        h_pool_flat = self.dropout(torch.reshape(h_pool, [-1, self.num_filter_total]))

        output = self.Weight(h_pool_flat) + self.bias  # [bs, n_class]

        return output


class DistilBERTAndTextCnnForSeq(DistilBertPreTrainedModel):
    def __init__(self, config, filters, filter_num):
        super(DistilBERTAndTextCnnForSeq, self).__init__(config)
        self.out_channels = 32
        # 获得预训练模型的参数
        self.config = DistilBertConfig(config)
        # 标签数量
        self.num_labels = 31
        # albert模型
        self.model = DistilBertModel(config)

        self.relu = nn.ReLU()
        self.filters = filters
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            return_dict=return_dict,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states  # shape = (batch_size=16, sequence_length=512, hidden_size=768)
        # hidden_states = outputs[0]
        cls_embeddings = hidden_states[0][:, 0, :].unsqueeze(1)
        # cls_embeddings = self.relu(hidden_states)
        for i in range(1, 7):
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


class Deberta(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super(Deberta, self).__init__(config)
        self.out_channels = 32
        # 获得预训练模型的参数
        self.config = DebertaV2Config(config)
        # 标签数量
        self.num_labels = 31
        # albert模型
        self.model = DebertaV2Model(config)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        # 最后的分类层
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # 初始化权重
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取模型结果
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            return_dict=return_dict,
            output_hidden_states=False
        )

        hidden_states = outputs.last_hidden_state  # shape = (batch_size=16, sequence_length=512, hidden_size=768)

        cls_embeddings = hidden_states[:, 0, :]

        # logits会返回一个还未经过
        logits = self.classifier(self.dropout(cls_embeddings))
        # logits = self.classifier(cls_embeddings)

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

