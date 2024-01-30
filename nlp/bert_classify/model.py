import torch.nn as nn
from transformers import BertModel
from parsers import parsers
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.args = parsers()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 加载预训练模型
        self.bert = BertModel.from_pretrained(self.args.bert_pred)
        # 打开bert梯度,方便后续微调
        for param in self.bert.parameters():
            param.requires_grad = True
        # 全连接层
        self.linear = nn.Linear(self.args.num_filters, self.args.class_num)

    def forward(self, x):
        input_ids, attention_mask = x[0].to(self.device), x[1].to(self.device)
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)
        pred = self.linear(hidden_out.pooler_output)
        #返回预测结果
        return pred


