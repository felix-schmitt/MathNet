UNK_THR = 1

train_file = open("../data/im2latex/im2latex_train_filter.lst", 'r').readlines()

formulas = open("../data/im2latex/im2latex_formulas.norm.lst", encoding="utf8", mode='r').readlines()
formulas = ["_START_ " + line.split('\n')[0] + " _END_" for line in formulas]
vocab_counts = {}
for line in train_file:
    img_name, formula_idx = line.split('\n')[0].split(' ')
    words = formulas[int(formula_idx)].split(' ')
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
latex_vocab = sorted(latex_vocab)

VOCAB_SIZE = len(latex_vocab)
print("vocab size: ", VOCAB_SIZE)
with open('../data/im2latex/latex.vocab', 'w') as f:
    for v in latex_vocab:
        f.write(v)
        f.write('\n')
