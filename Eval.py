import torch
import torch.nn as nn

# Initial consonants (초성)
CHOSEONG = [
    'ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ',
    'ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'
]

# Medial vowels (중성)
JUNGSEONG = [
    'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ',
    'ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'
]

# Final consonants (종성)
JONGSEONG = [
    'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
    'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ',
    'ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'
]

def decompose_hangul_flat(text: str):
    JONGSEONG = [
        '', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
        'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ',
        'ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'
    ]

    result = []

    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            s_index = code - 0xAC00
            result.append(CHOSEONG[s_index // 588])
            result.append(JUNGSEONG[(s_index % 588) // 28])
            jong = JONGSEONG[s_index % 28]
            if jong:
                result.append(jong)
        else:
            result.append(ch)

    return result

# Full vocabulary
JAMO_VOCAB = CHOSEONG + JUNGSEONG + JONGSEONG
JAMO_VOCAB = sorted(list(set(JAMO_VOCAB)))
JAMO_TO_INDEX = {jamo: i for i, jamo in enumerate(JAMO_VOCAB)}
VOCAB_SIZE = len(JAMO_VOCAB)

HIRAGANA_VOCAB = [
    'ぁ','あ','ぃ','い','ぅ','う','ぇ','え','ぉ','お',
    'か','が','き','ぎ','く','ぐ','け','げ','こ','ご',
    'さ','ざ','し','じ','す','ず','せ','ぜ','そ','ぞ',
    'た','だ','ち','ぢ','っ','つ','づ','て','で','と','ど',
    'な','に','ぬ','ね','の',
    'は','ば','ぱ','ひ','び','ぴ','ふ','ぶ','ぷ','へ','べ','ぺ','ほ','ぼ','ぽ',
    'ま','み','む','め','も',
    'ゃ','や','ゅ','ゆ','ょ','よ',
    'ら','り','る','れ','ろ',
    'ゎ','わ','ゐ','ゑ','を','ん',
    'ー'
]

HIRAGANA_TO_INDEX = {c: i for i, c in enumerate(HIRAGANA_VOCAB)}
HIR_VOCAB_SIZE = len(HIRAGANA_VOCAB)

JAMO_VOCAB_SIZE = len(JAMO_VOCAB) + 3   # + PAD, SOS, EOS
JP_VOCAB_SIZE   = len(HIRAGANA_VOCAB) + 3     # Hiragana / Katakana / Kanji vocab

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (batch, src_len)
        emb = self.embedding(x)
        _, (h, c) = self.lstm(emb)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x: (batch, 1)
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)

        hidden = self.encoder(src)
        x = tgt[:, 0].unsqueeze(1)  # <SOS>

        for t in range(1, tgt_len):
            out, hidden = self.decoder(x, hidden)
            outputs[:, t] = out.squeeze(1)

            teacher = torch.rand(1).item() < teacher_forcing_ratio
            x = tgt[:, t].unsqueeze(1) if teacher else out.argmax(-1)

        return outputs
    
EMBED_DIM  = 256
HIDDEN_DIM = 512

encoder = Encoder(
    vocab_size=JAMO_VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
)

decoder = Decoder(
    vocab_size=JP_VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
)

model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load("seq2seq_final.pt"))

JAMO_TO_ID = {j: i + 3 for i, j in enumerate(JAMO_VOCAB)}
JP_TO_ID = {c: i + 3 for i, c in enumerate(HIRAGANA_VOCAB)}
ID_TO_JAMO = {i: j for j, i in JAMO_TO_ID.items()}
ID_TO_JP = {i: c for c, i in JP_TO_ID.items()}

MAX_LEN = 12          # max jamo length (pad/truncate)
MAX_JAPANESE_LEN = 11 # max japanese length (pad/truncate)

def encode_jamo(jamo_list, max_len):
    ids = [SOS_ID]

    for j in jamo_list:
        ids.append(JAMO_TO_ID.get(j, PAD_ID))

    ids.append(EOS_ID)

    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = EOS_ID

    return ids

def encode_japanese(text, max_len):
    ids = [SOS_ID]

    for ch in text:
        ids.append(JP_TO_ID.get(ch, PAD_ID))

    ids.append(EOS_ID)

    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = EOS_ID

    return ids

def evaluate_once(model, jamo_list, max_src_len, max_tgt_len):
    model.eval()

    # Encode source
    src_ids = encode_jamo(jamo_list, max_src_len)
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)

    result_ids = []

    with torch.no_grad():
        # Encode
        hidden = model.encoder(src)

        # Start with <SOS>
        x = torch.tensor([[SOS_ID]], dtype=torch.long)

        for _ in range(max_tgt_len):
            out, hidden = model.decoder(x, hidden)
            token_id = out.argmax(-1).item()

            if token_id == EOS_ID:
                break

            result_ids.append(token_id)
            x = torch.tensor([[token_id]], dtype=torch.long)

    # Convert IDs → string
    result = ''.join(ID_TO_JP[i] for i in result_ids if i in ID_TO_JP)

    return result

while 1:
    x = input("한국어를 입력하세요: ")
    if x == 'exit':
        break
    print("변환된 일본어 발음: ", end="")
    print(evaluate_once(model, decompose_hangul_flat(x), MAX_LEN, MAX_JAPANESE_LEN))