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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn

# Импорт параметров из config.py
from config import (
    DATA_PATH, MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, PATIENCE,
    EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT,
    LEARNING_RATE, BETAS, EPS, VOCAB_SIZE, MAX_LEN, SEED
)

# Фиксация random seed для воспроизводимости
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Конфигурация устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка данных
def load_and_preprocess_data(file_path):
    """
    Загружает и предобрабатывает данные для машинного перевода.
    Очищает пунктуацию, приводит к нижнему регистру, добавляет специальные токены.
    Возвращает список пар (английский, русский).
    """
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

# Проверка наличия данных
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}")

all_pairs = load_and_preprocess_data(DATA_PATH)
print(f"Загружено пар: {len(all_pairs)}")

# Словари
class Vocabulary:
    """
    Класс для построения и хранения словаря токенов.
    Позволяет преобразовывать слова в индексы и обратно.
    """
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.size = 2
    
    def build(self, texts, max_size=30000, update=False):
        """
        Строит словарь по списку текстов.
        Аргументы:
            texts: список строк
            max_size: максимальный размер словаря
            update: если True — добавляет новые слова к существующему словарю
        """
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
                 embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT, max_len=MAX_LEN):
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
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_LEN
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Загрузка существующей модели и истории обучения
start_epoch = 0
best_loss = float('inf')
epochs_no_improve = 0
train_losses, val_losses, val_bleus = [], [], []

if os.path.exists(MODEL_PATH):
    print("\nЗагрузка модели для дообучения...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint.get('best_loss', float('inf'))
    start_epoch = checkpoint.get('epoch', 0) + 1
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    val_bleus = checkpoint.get('val_bleus', [])
    print(f"Восстановлено с эпохи {start_epoch}, best_loss={best_loss:.4f}")

# Обучение

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

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len=MAX_LEN, device="cpu"):
    model.eval()
    src = src.unsqueeze(0) if src.dim() == 1 else src
    src_pos = torch.arange(0, src.size(1), device=device).unsqueeze(0)
    src_emb = model.encoder_embed(src) + model.pos_encoder(src_pos)
    memory = model.transformer.encoder(src_emb.permute(1,0,2))
    ys = torch.tensor([[tgt_vocab.word2idx.get("[start]", 1)]], device=device)
    for i in range(max_len):
        tgt_pos = torch.arange(0, ys.size(1), device=device).unsqueeze(0)
        tgt_emb = model.decoder_embed(ys) + model.pos_encoder(tgt_pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.transformer.decoder(
            tgt_emb.permute(1,0,2), memory,
            tgt_mask=tgt_mask
        )
        out = model.fc_out(out.permute(1,0,2))[:, -1, :]
        prob = out.argmax(dim=-1)
        ys = torch.cat([ys, prob.unsqueeze(0)], dim=1)
        if prob.item() == tgt_vocab.word2idx.get("[end]", 1):
            break
    return ys.squeeze().tolist()

def ids_to_text(ids, vocab):
    words = []
    for idx in ids:
        word = vocab.idx2word.get(idx, "<unk>")
        if word in ["[start]", "[end]", "<pad>"]:
            continue
        words.append(word)
    return words

def validate():
    model.eval()
    total_loss = 0
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Validation"):
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            # BLEU
            for i in range(src.size(0)):
                pred_ids = greedy_decode(model, src[i], src_vocab, tgt_vocab, max_len=MAX_LEN, device=device)
                ref_ids = tgt[i].tolist()
                pred_words = ids_to_text(pred_ids, tgt_vocab)
                ref_words = [ids_to_text(ref_ids, tgt_vocab)]
                if len(pred_words) > 0 and len(ref_words[0]) > 0:
                    bleu = sentence_bleu(ref_words, pred_words, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    return total_loss / len(val_loader), avg_bleu

# TensorBoard writer
writer = SummaryWriter(log_dir="runs")

# Красивый прогресс-бар и сохранение графика
def save_plot(train_losses, val_losses, val_bleus, epoch):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Progress")
    plt.subplot(1,2,2)
    plt.plot(range(1, len(val_bleus)+1), val_bleus, label="Val BLEU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.legend()
    plt.title("BLEU Progress")
    plt.tight_layout()
    plt.savefig(f"progress_epoch_{epoch+1}.png")
    plt.close()

writer = SummaryWriter(log_dir="runs")

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    "[",
    TimeElapsedColumn(),
    "<",
    TimeRemainingColumn(),
    "]",
    transient=True
) as progress:
    train_task = progress.add_task("Обучение", total=NUM_EPOCHS)

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_epoch()
        val_loss, val_bleu = validate()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {val_bleu:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch+1)
        writer.add_scalar("Loss/Val", val_loss, epoch+1)
        writer.add_scalar("BLEU/Val", val_bleu, epoch+1)

        if (epoch+1) % 2 == 0:
            save_plot(train_losses, val_losses, val_bleus, epoch)

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
                },
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_bleus': val_bleus
            }, MODEL_PATH)
            print(f"Модель сохранена!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Ранняя остановка!")
                break
        progress.update(train_task, advance=1)

writer.close()
