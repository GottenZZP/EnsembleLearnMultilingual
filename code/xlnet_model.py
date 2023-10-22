from process import InputDataSet
from transformers import DebertaV2Model, DebertaV2Config, DebertaV2PreTrainedModel
from transformers import XLNetModel, XLNetConfig, XLNetPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging
from torch import nn
from torch.nn import functional as F
import torch

logging.set_verbosity_error()


class XLNet(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNet, self).__init__(config)
        self.out_channels = 32
        # 获得预训练模型的参数
        self.config = XLNetConfig(config)
        # 标签数量
        self.num_labels = 31
        # albert模型
        self.model = XLNetModel(config)

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

