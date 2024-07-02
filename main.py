import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダー
class VQADataset(torch.utils.data.Dataset): # torchのDatasetの作成をする
  def __init__(self, df_path, image_dir, transform=None, answer=True):
    self.transform = transform # 画像の前処理
    self.image_dir = image_dir # 画像のフォルダ
    self.df = pd.read_json(df_path) # json -> df
    self.answer = answer # 答え

    answer_copus = pd.read_csv("./data/data_annotations_class_mapping.csv")
    self.answer2idx = dict(zip(answer_copus["answer"], answer_copus["class_id"]))
    self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    if self.answer:
      for answers in self.df["answers"]:
        for answer in answers:
          word = answer["answer"]
          word = process_text(word)
          if word not in self.answer2idx:
            self.answer2idx[word] = len(self.answer2idx)
      self.idx2answer = {v: k for k, v in self.answer2idx.items()}

  def update_dict(self, dataset):
      self.answer2idx = dataset.answer2idx
      self.idx2answer = dataset.idx2answer

  def __getitem__(self, idx):
    image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
    image = self.transform(image) 
    question = process_text(self.df["question"][idx])


    if self.answer:
      answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
      mode_answer_idx = mode(answers)
      return image, question, torch.Tensor(answers), int(mode_answer_idx)
    else:
      return image, question

  def __len__(self):
    return len(self.df)


# 2. 評価指標の実装
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
from transformers import BertTokenizer, BertModel,AutoProcessor, CLIPVisionModel

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", return_tensors="pt", padding=True)
        self.bert_model = BertModel.from_pretrained("bert-large-uncased", torch_dtype=torch.float32, attn_implementation="sdpa")
        for param in self.bert_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1024+1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        with torch.no_grad():
          inputs = self.processor(images=image, return_tensors="pt", do_rescale=False).to(image.device)
          image_feature = self.clip_model(**inputs).last_hidden_state[:, 0, :]
          question = self.bert_tokenizer(question,truncation=True, padding=True, return_tensors="pt").to(image.device)
          question_feature = self.bert_model(**question).last_hidden_state[:, 0, :]

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
