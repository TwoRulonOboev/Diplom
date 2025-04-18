import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
import string
import random
import os
from tqdm import tqdm

# Конфигурация устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка данных
def load_and_preprocess_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]
        
        pairs = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                english = re.sub(f"[{re.escape(string.punctuation)}]", "", parts[0].lower())
                russian = "[start] " + re.sub(f"[{re.escape(string.punctuation)}]", "", parts[1].lower()) + " [end]"
                pairs.append((english, russian))
        
        random.shuffle(pairs)
        return pairs
    
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return []

# Пути и параметры
DATA_PATH = "data/rus.txt"  # Новый датасет
MODEL_PATH = "best_model.pth"
MAX_LEN = 25
BATCH_SIZE = 64

# Проверка наличия данных
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}")

all_pairs = load_and_preprocess_data(DATA_PATH)
print(f"Загружено пар: {len(all_pairs)}")

# Словари
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.size = 2
    
    def build(self, texts, max_size=30000, update=False):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for word, _ in sorted_words:
            if word not in self.word2idx and (update or self.size < max_size):
                self.idx2word[self.size] = word
                self.word2idx[word] = self.size
                self.size += 1

# Загрузка или создание словарей
if os.path.exists(MODEL_PATH):
    # Загрузка существующих словарей из модели
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    src_vocab = Vocabulary()
    src_vocab.__dict__ = checkpoint['src_vocab']
    tgt_vocab = Vocabulary()
    tgt_vocab.__dict__ = checkpoint['tgt_vocab']
    print("\nЗагружены существующие словари.")
else:
    # Создание новых словарей
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    print("\nСозданы новые словари.")

# Обновление словарей новыми данными
src_texts = [pair[0] for pair in all_pairs]
tgt_texts = [pair[1] for pair in all_pairs]
src_vocab.build(src_texts, update=True)  # Режим обновления
tgt_vocab.build(tgt_texts, update=True)

print(f"\nРазмеры словарей:")
print(f"Английский: {src_vocab.size}, Русский: {tgt_vocab.size}")

# Датасет и DataLoader
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        
        # Обработка пустых строк
        if not src.strip() or not tgt.strip():
            return self.__getitem__(random.randint(0, len(self.pairs)-1))
        
        src_ids = [self.src_vocab.word2idx.get(word, 1) for word in src.split()[:MAX_LEN]]
        tgt_ids = [self.tgt_vocab.word2idx.get(word, 1) for word in tgt.split()[:MAX_LEN+1]]
        
        # Проверка на выход за границы
        if max(src_ids) >= self.src_vocab.size or max(tgt_ids) >= self.tgt_vocab.size:
            return self.__getitem__(random.randint(0, len(self.pairs)-1))
            
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded.to(device), tgt_padded.to(device)

# Разделение данных
train_size = int(0.8 * len(all_pairs))
val_size = len(all_pairs) - train_size
train_loader = DataLoader(
    TranslationDataset(all_pairs[:train_size], src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True
)
val_loader = DataLoader(
    TranslationDataset(all_pairs[train_size:], src_vocab, tgt_vocab),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# Модель
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_dim=256, num_heads=8, ff_dim=2048, 
                 num_layers=3, dropout=0.1, max_len=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(max_len, embed_dim)
        
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_pos = torch.arange(0, src.size(1), device=device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=device).unsqueeze(0)
        
        src_emb = self.encoder_embed(src) + self.pos_encoder(src_pos)
        tgt_emb = self.decoder_embed(tgt) + self.pos_encoder(tgt_pos)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        output = self.transformer(
            src_emb.permute(1,0,2),
            tgt_emb.permute(1,0,2),
            tgt_mask=tgt_mask
        )
        return self.fc_out(output.permute(1,0,2))

# Инициализация модели
model = TransformerTranslator(
    src_vocab_size=src_vocab.size,
    tgt_vocab_size=tgt_vocab.size,
    max_len=MAX_LEN
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Загрузка существующей модели
if os.path.exists(MODEL_PATH):
    print("\nЗагрузка модели для дообучения...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Обучение
NUM_EPOCHS = 50
PATIENCE = 3
best_loss = float('inf')
epochs_no_improve = 0

def train_epoch():
    model.train()
    total_loss = 0
    for src, tgt in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Validation"):
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch()
    val_loss = validate()
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'src_vocab': src_vocab.__dict__,
            'tgt_vocab': tgt_vocab.__dict__,
            'model_params': {
                'embed_dim': model.embed_dim,
                'num_heads': model.num_heads,
                'ff_dim': model.ff_dim,
                'num_layers': model.num_layers,
                'dropout': model.dropout,
                'max_len': MAX_LEN
            }
        }, MODEL_PATH)
        print(f"Модель сохранена!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Ранняя остановка!")
            break