import torch
import re
import string

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

class TransformerTranslator(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=256, num_heads=8, ff_dim=2048,
                 num_layers=3, dropout=0.1, max_len=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len

        self.encoder_embed = torch.nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embed = torch.nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = torch.nn.Embedding(max_len, embed_dim)

        self.transformer = torch.nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.fc_out = torch.nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)

        src_emb = self.encoder_embed(src) + self.pos_encoder(src_pos)
        tgt_emb = self.decoder_embed(tgt) + self.pos_encoder(tgt_pos)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        output = self.transformer(
            src_emb.permute(1, 0, 2),  # (seq_len, batch, features)
            tgt_emb.permute(1, 0, 2),  # (seq_len, batch, features)
            tgt_mask=tgt_mask
        )
        return self.fc_out(output.permute(1, 0, 2))  # (batch, seq_len, vocab)

def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)

    src_vocab = Vocabulary()
    src_vocab.__dict__ = checkpoint['src_vocab']
    tgt_vocab = Vocabulary()
    tgt_vocab.__dict__ = checkpoint['tgt_vocab']

    params = checkpoint['model_params']
    model = TransformerTranslator(
        src_vocab_size=src_vocab.size,
        tgt_vocab_size=tgt_vocab.size,
        **params
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, src_vocab, tgt_vocab, params

def preprocess(text, vocab, max_len=25):
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text.lower())
    words = text.split()[:max_len]
    return [vocab.word2idx.get(word, 1) for word in words]

def translate(model, src_vocab, tgt_vocab, text, device='cpu', max_len=25):
    model.to(device)
    src = torch.LongTensor(preprocess(text, src_vocab, max_len)).unsqueeze(0).to(device)

    output = [tgt_vocab.word2idx['[start]']]
    for _ in range(max_len):
        tgt = torch.LongTensor(output).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(src, tgt)
        next_idx = pred[0, -1].argmax().item()  # Исправлено здесь!
        if next_idx == tgt_vocab.word2idx['[end]']:
            break
        output.append(next_idx)

    translated = ' '.join([tgt_vocab.idx2word.get(idx, '<unk>') for idx in output])
    return translated.replace('[start]', '').replace('[end]', '').strip()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, src_vocab, tgt_vocab, params = load_model('data/best_model.pth', device)

    print("Модель загружена! Введите текст для перевода (или 'exit' для выхода).")
    while True:
        text = input("\nВведите английский текст: ")
        if text.lower() == 'exit':
            break
        translation = translate(model, src_vocab, tgt_vocab, text, device)
        print(f"Перевод: {translation}")