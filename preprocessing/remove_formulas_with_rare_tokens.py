import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

UNK_THR = 0
dataset_folder = "data/61styles-v4/"
train_file = "61styles-v4-train.csv"
vocab_file = "data/vocabs/latex_v4.vocab"
csv_files = ["61styles-v4-train.csv", "61styles-v4-validate.csv", "61styles-v4-test.csv"]

vocab_counts = {}
with open(dataset_folder + train_file, 'r') as train_file:
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
vocab_contribution = {}
for key, value in vocab_counts.items():
    if value in vocab_contribution:
        vocab_contribution[value] += 1
    else:
        vocab_contribution[value] = 1
print(dict(sorted(vocab_contribution.items())))
remove_tokens = [key for key, value in vocab_counts.items() if value < 20]
print(remove_tokens)
for file in csv_files:
    removed = 0
    with open(dataset_folder + file) as f:
        reader = csv.reader(f)
        with open(dataset_folder + file.replace(".csv", "_no_rare_tokens.csv"), 'w') as ff:
            writer = csv.writer(ff)
            for line in tqdm(reader):
                formula = line[0].split(" ")
                if not any([remove_token in formula for remove_token in remove_tokens]):
                    writer.writerow(line)
                else:
                    removed += 1
    print(f"{file} removed {removed} formulas")

