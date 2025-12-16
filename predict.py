import torch
from nets import ALBERT
from transformers import BertTokenizer

device = torch.device("cuda")
model = ALBERT(num_classes=7)
model.load_state_dict(torch.load("models/albert_best.pth"))
model = model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("albert-chinese-small")


def predict_res(s):
    info_dict = tokenizer(s, max_length=200, truncation=True, padding="max_length")
    input_ids = torch.unsqueeze(torch.LongTensor(info_dict["input_ids"]), dim=0).to(device)
    attention_mask = torch.unsqueeze(torch.LongTensor(info_dict["attention_mask"]), dim=0).to(device)
    label_names = ['like', 'sadness', 'fear', 'anger', 'disgust', 'happiness', 'surprise']
    with torch.no_grad():
        index = torch.argmax(model(input_ids, attention_mask), dim=-1).cpu().numpy()[0]

    res = label_names[index]

    print(f"【今天不是个好日子】的预测结果为:[{res}]")


if __name__ == '__main__':
    predict_res("")

