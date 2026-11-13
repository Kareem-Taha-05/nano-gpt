import re
from collections import Counter
from itertools import islice
import numpy as np

# one shared pre-tokenizer for the word and bpe tokenizers: a word together with its leading space, a single punctuation mark, or a run of whitespace
# keeping the space glued to the word means decode is just ''.join(...) and nothing is ever lost
PRETOKEN = re.compile(r" ?\w+| ?[^\w\s]|\s+")


class CharTokenizer:
    """ every character is a token, this is exactly what we built at the start of the notebook """

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.word2num = {ch: i for i, ch in enumerate(chars)}
        self.num2word = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        # lookup table indexed by unicode code point, lets us encode hundreds of MB with numpy instead of a python loop
        self.lut = np.full(max(ord(c) for c in chars) + 1, -1, dtype=np.int32)
        for c, i in self.word2num.items():
            self.lut[ord(c)] = i

    def encode(self, s):
        return [self.word2num[c] for c in s]

    def encode_np(self, s):
        cp = np.frombuffer(s.encode('utf-32-le'), dtype=np.uint32)  # the code point of every character, in one go
        if len(cp) and int(cp.max()) >= len(self.lut):
            raise KeyError("character not in vocabulary")
        ids = self.lut[cp]
        if (ids < 0).any():
            raise KeyError("character not in vocabulary")
        return ids

    def decode(self, ids):
        return ''.join(self.num2word[i] for i in ids)


class WordTokenizer:
    """ every (space + word) is a token, rare words collapse into <unk> so the vocabulary doesn't explode """

    def __init__(self, text, min_freq=3):
        counts = Counter(PRETOKEN.findall(text))
        words = ['<unk>'] + sorted(w for w, c in counts.items() if c >= min_freq)  # <unk> sits at index 0
        self.word2num = {w: i for i, w in enumerate(words)}
        self.num2word = {i: w for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def encode(self, s):
        return [self.word2num.get(w, 0) for w in PRETOKEN.findall(s)]  # anything we haven't seen before maps to <unk>

    def decode(self, ids):
        return ''.join(self.num2word[i] for i in ids)


class BPETokenizer:
    """ byte pair encoding: start from raw bytes and repeatedly merge the most frequent neighbouring pair into a new token """

    def __init__(self, text, num_merges=256):
        word_counts = Counter(PRETOKEN.findall(text))
        # merges never cross word boundaries, so we only need to train on the unique words weighted by how often they appear
        # (this makes training seconds instead of minutes in pure python)
        words = {w: list(w.encode('utf-8')) for w in word_counts}
        self.merges = {}  # (id_a, id_b) -> new id, the order of insertion is the order of priority
        self.vocab = {i: bytes([i]) for i in range(256)}
        for n in range(num_merges):
            pairs = Counter()
            for w, ids in words.items():
                for p in zip(ids, ids[1:]):
                    pairs[p] += word_counts[w]
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + n
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in words:
                words[w] = self._merge(words[w], best, new_id)
        self.vocab_size = 256 + len(self.merges)
        self.cache = {}

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_word(self, w):
        if w in self.cache:
            return self.cache[w]
        ids = list(w.encode('utf-8'))
        while len(ids) >= 2:
            # always apply the merge that was learned first (lowest id), that is what makes the encoding deterministic
            best = min(set(zip(ids, ids[1:])), key=lambda p: self.merges.get(p, float('inf')))
            if best not in self.merges:
                break
            ids = self._merge(ids, best, self.merges[best])
        self.cache[w] = ids
        return ids

    def encode(self, s):
        return [i for w in PRETOKEN.findall(s) for i in self._encode_word(w)]

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')


def _split_at_newlines(text, size=1_000_000):
    # ~1MB pieces, always cut right after a newline so no word is ever split in half
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.find("\n", j)
            j = len(text) if k == -1 else k + 1
        yield text[i:j]
        i = j


class HFBPETokenizer:
    """ the same byte-level bpe idea, but trained and run by huggingface's rust `tokenizers` library (pip install tokenizers) """

    def __init__(self, text, num_merges=256):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers  # imported here so the project works without it
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # bytes first, exactly like our own implementation, so nothing is ever unknown
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=256 + num_merges, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[], show_progress=False)
        tok.train_from_iterator(_split_at_newlines(text), trainer)
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode_np(self, s):
        # encode_batch hands back every token id as a python int, ~30 bytes each, so on a few hundred MB of text we must convert to numpy (4 bytes each) in small groups
        # instead of holding all of it at once, otherwise the notebook runs out of ram
        pieces = _split_at_newlines(s)
        out = []
        while True:
            group = list(islice(pieces, 32))  # 32 pieces of ~1MB
            if not group:
                break
            out.extend(np.array(e.ids, dtype=np.int32) for e in self.tok.encode_batch(group))  # encode_batch runs the pieces in parallel on all cpu cores
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)

    def encode(self, s):
        return self.encode_np(s).tolist()

    def decode(self, ids):
        return self.tok.decode(list(ids))

    # the rust object can't be pickled directly, so the checkpoint stores it as a json string and rebuilds it on load
    def __getstate__(self):
        return {'json': self.tok.to_str(), 'vocab_size': self.vocab_size}

    def __setstate__(self, state):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_str(state['json'])
        self.vocab_size = state['vocab_size']


def get_tokenizer(cfg, text):
    if cfg.tokenizer == "char":
        return CharTokenizer(text)
    if cfg.tokenizer == "word":
        return WordTokenizer(text, cfg.word_min_freq)
    if cfg.bpe_backend == "hf":
        return HFBPETokenizer(text, cfg.bpe_merges)
    return BPETokenizer(text, cfg.bpe_merges)
