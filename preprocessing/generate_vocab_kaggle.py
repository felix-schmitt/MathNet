import csv

UNK_THR = 0
dataset_folder = "data/im2latexv2styles/"
train_files = ["data/im2latexv2styles/im2latexv2styles-train_normalized.csv"]
vocab_file = "data/vocabs/im2latexv2styles.vocab"

vocab_counts = {}
for train_file in train_files:
    with open(train_file, 'r') as train_file:
        train_lines = csv.reader(train_file)
        for i, line in enumerate(train_lines):
            formula = line[0]
            if i == 0:
                continue
            words = formula.split(' ')
            for w in words:
                if w not in vocab_counts.keys():
                    vocab_counts[w] = 1
                else:
                    vocab_counts[w] += 1
latex_vocab = set()
for w in vocab_counts.keys():
    if w == '':
        continue
    if vocab_counts[w] <= UNK_THR:
        pass
    else:
        latex_vocab.add(w)

print("Unk counts: ", len(vocab_counts.keys()) - len(latex_vocab))
latex_vocab.add('_PAD_')
latex_vocab.add('_UNK_')
latex_vocab.add('_START_')
latex_vocab.add('_END_')
latex_vocab = sorted(latex_vocab)

VOCAB_SIZE = len(latex_vocab)
print("vocab size: ", VOCAB_SIZE)
with open(vocab_file, 'w') as f:
    for v in latex_vocab:
        f.write(v)
        f.write('\n')
