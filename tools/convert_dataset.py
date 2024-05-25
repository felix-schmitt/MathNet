import csv

def main(input_files):
    formulas = read_file(input_files['formulas'])
    process_file(input_files['train'], formulas)
    process_file(input_files['val'], formulas)
    process_file(input_files['test'], formulas)


def read_file(file):
    temp = []
    with open(file) as f:
        temp_line = f.readline()
        while temp_line:
            temp.append(temp_line)
            temp_line = f.readline()
    f.close()
    return temp

def process_file(file, formulas):
    data = read_file(file)
    csv_file_name = file + ".csv"
    with open(csv_file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["formula", "image"])
        for line in data:
            temp_line = line.split(' ')
            writer.writerow([formulas[int(temp_line[1].replace("\n", ""))][:-1], temp_line[0]])


if __name__ == '__main__':
    input_files = {'formulas': "data/im2latex/im2latex_formulas.norm.lst",
                   'train': "data/im2latex/im2latex_train_filter.lst",
                   'val': "data/im2latex/im2latex_validate_filter.lst",
                   'test': "data/im2latex/im2latex_test_filter.lst"}
    main(input_files)