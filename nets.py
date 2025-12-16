import torch
import torch.nn as nn
from transformers import AlbertModel


class ALBERT(nn.Module):

    def __init__(self, num_classes=2):
        super(ALBERT, self).__init__()
        self.bert = AlbertModel.from_pretrained("albert-chinese-small")
        self.linear = nn.Sequential(
            nn.Linear(384, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        _, out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return self.linear(out)


if __name__ == '__main__':
    x = torch.ones((4, 300)).long()
    m = torch.ones((4, 300)).long()
    model = ALBERT()
    print(model(x, m).size())
