import csv
import shutil
import copy
from typing import Tuple
from pathlib import Path
import preprocessing.config_latex as config_latex
import regex as re
import arxiv
from tqdm import tqdm
import tarfile
import time

class tokenize():
    def __init__(self, csv_file, output_file):
        self.csv_file = csv_file
        self.output_file = output_file
        self.tokenized_formulae = []
        self.stop_elements = ['_', '~', '^', '{', '}', '(']

    def process_file(self):
        with open(self.csv_file) as f:
            with open(self.output_file, "w") as out_f:
                formula = f.readline()
                writer = csv.writer(out_f)
                writer.writerow(["formula", "image"])
                while formula:
                    tokens = self._tokenize(formula[:-1])
                    writer.writerow([" ".join(tokens), ""])
                    formula = f.readline()

    def _tokenize(self, formula):
        tokens = []
        token = ''
        if formula[0] == "$" and formula[-1] == "$":
            formula = formula[1:-1]
        for element_i, element in enumerate(list(formula)):
            if element == " ":
                if token:
                    tokens.append(token)
                    token = ''
                continue
            if not token and element != '\\':
                tokens.append(element)
                continue
            elif not token:
                token += element
                continue
            if token == '\\' and element == '\\':
                token += element
                tokens.append(token)
                token = ''
                continue
            if element in self.stop_elements:
                tokens.append(token)
                token = ''
                if element == '\\':
                    token += element
                else:
                    tokens.append(element)
                continue
            else:
                token += element
        if token:
            tokens.append(token)
        return tokens

class collect():
    def __init__(self, temp_folder="temp",
                 arxiv_args={'query': "cs.AI", 'max_results': 100, 'sort_by': arxiv.SortCriterion.SubmittedDate},
                 output_file="data/new_datapoints/inline-1.csv", track_file="data/new_datapoints/tracking_file_inline.csv",
                 math_type=['inline']):
        self.temp_folder = Path(temp_folder)
        self.arxiv_args = arxiv_args
        self.papers = list(arxiv.Search(**self.arxiv_args).results())
        self.output_file = output_file
        (Path(self.output_file)).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.output_file).exists():
            with open(self.output_file, "w") as f:
                print(f"created track file {self.output_file}")
        self.track_file = track_file
        if not Path(self.track_file).exists():
            with open(self.track_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(['paper id'])
                print(f"created track file {self.track_file}")
        self.math_type = math_type

    def extract_formulas(self):
        start_time = 0
        with tqdm(self.papers, postfix="download images") as pbar:
            for paper in pbar:
                if self.temp_folder.exists():
                    shutil.rmtree(str(self.temp_folder))
                pbar.set_description(f"download {paper.entry_id}")
                # check that file was not processed before
                with open(self.track_file) as f:
                    reader = list(csv.reader(f))
                    if [paper.entry_id[:-2]] in reader:
                        continue
                with open(self.track_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([paper.entry_id[:-2]])
                # download paper
                Path("temp").mkdir()
                wait_time = start_time - time.time() + 5
                if wait_time > 0:
                    time.sleep(wait_time)
                start_time = time.time()
                if not paper.download_source(dirpath=self.temp_folder, filename="temp.tar.gz"):
                    print(f"Could not download {paper.entry_id}")
                tar = tarfile.open(self.temp_folder / "temp.tar.gz", "r:gz")
                tar.extractall(self.temp_folder)
                tar.close()
                # rename largest tex file
                tex_files = sorted(self.temp_folder.glob('**/*.tex'))
                for tex_file in tex_files:
                    try:
                        data = Path(tex_file).read_text()
                    except UnicodeDecodeError:
                        continue
                    new_latex, positions = self.get_math_positions(data)
                    for formula_type, position in positions.items():
                        if not formula_type in self.math_type:
                            continue
                        with open(self.output_file, "a") as f:
                            for start_i, stop_i in position:
                                f.write(new_latex[start_i:stop_i]+"\n")

    def process_environment(self, data: str, search_pattern: str, env_type: str, ignore_pos: list = None) -> Tuple[
        list, str]:
        """Process LaTeX environment."""

        try:
            start_pos = re.search(r"\\begin\{document\}", data).end()
        except AttributeError:
            start_pos = 0
        if env_type == 'inline':
            start_pos = 0
        vertices_ptrs = []
        ignore_pos = ignore_pos or []
        regex = re.compile(search_pattern)
        start_pattern = False
        if ").*?" in search_pattern and env_type != 'display':
            start_pattern = search_pattern[9:search_pattern.find(").*?")].replace("\\{", "{")
        end_pattern = False
        if env_type == 'inline' or env_type == 'footnote':
            end_pattern = '$'
        if search_pattern[search_pattern.find('.*?(?=') + 6:-1] == '\\}':
            end_pattern = '}'
        while start_pos <= len(data):
            pattern = regex.search(data, pos=start_pos)
            # if found a pattern
            if pattern:
                # get start end position
                start, end = pattern.span()
                if (env_type == 'display_lyx' or env_type == 'inline_lyx') and data[start - 3:start - 2] == '\\':
                    start_pos = start + 1
                    continue
                if (env_type == 'display_lyx' or env_type == 'inline_lyx') and data[end - 1:end] == '\\':
                    end_pattern_found = False
                    while not end_pattern_found:
                        end += 2
                        if env_type == 'inline_lyx':
                            end_new = data[end:].find('\\)')
                        else:
                            end_new = data[end:].find('\\]')
                        end += end_new
                        if data[end - 1:end] != '\\':
                            end_pattern_found = True
                if env_type == 'caption2':
                    if end - start < 30:
                        start = end + 2
                        end = start + data[start:].find('}')
                    start += 1

                # check that no open brackets and $ exists
                while end_pattern:
                    open_brackets = len(
                        [match for match in re.finditer('{', data[start:end]) if
                         data[start + match.span()[0] - 1] != '\\'])
                    closed_brackets = len(
                        [match for match in re.finditer('}', data[start:end]) if
                         data[start + match.span()[0] - 1] != '\\'])
                    dollar_signs = len([match for match in re.finditer('\$', data[start:end]) if
                                        data[start + match.span()[0] - 1] != '\\'])
                    if open_brackets > closed_brackets or dollar_signs % 2 != 0:
                        end += 1
                        new_end = data[end:].find(end_pattern)
                        if new_end == -1:
                            break
                        if end_pattern == '$':
                            new_end += 1
                        end += new_end
                    else:
                        break
                if env_type == 'display':
                    start -= 2
                    end += 2
                found = pattern.group()
                # make further content checks
                skip = config_latex.ignore_content(found, env_type)
                # check if $ is a $ sign instead of the inline env
                if (env_type == 'inline' or env_type == 'display') and not skip:
                    # fixes $ signs in latex code
                    if data[start - 1:start + 1] == '\\$' and data[start - 2:start + 1] != '\\\\$':
                        start_pos = start + 2
                        continue
                # adds word before _123
                if 'inline' in env_type:
                    temp = copy.deepcopy(data[start:end])
                    temp = temp.replace('$', '').replace(" ", "").replace("{", '').replace("}", "")
                    if len(temp) >= 2 and temp[0] == '_' and data[start - 1] != ' ':
                        word_before = data[start - 10:start].split()[-1]
                        start -= len(word_before)

                # check ignore positions
                end_ignore_pos = False
                for pos in ignore_pos:
                    if pos[1] >= start >= pos[0]:
                        skip = True
                        end_ignore_pos = pos[1]
                        if start == end_ignore_pos:
                            end_ignore_pos = False
                        break
                if env_type == 'figure2':
                    start -= 1
                # check if definition of latex function
                if start_pattern:
                    before = data[start - 15 - len(start_pattern):start - len(start_pattern)]
                    before = before.replace(" ", "")
                    if """\\newcommand{""" in before or """\\renewcommand{""" in before or """\\def""" in before:
                        skip = True
                # if skip make no coloring
                if skip:
                    if end_ignore_pos:
                        start_pos = end_ignore_pos
                    elif found == '':
                        start_pos = end + 1
                    else:
                        start_pos = end
                    continue
                else:
                    vertices_ptrs.append((start, end))
                    start_pos = end
                    if 'footnote' in env_type and found == '' or start == end:
                        start_pos = end + 1
                    if env_type == 'display':
                        start_pos += 2
            else:
                break
        return vertices_ptrs

    def get_math_positions(self, data):
        # copy
        old_latex = copy.deepcopy(data)

        start_index = 0
        # modify header
        for index, row in enumerate(old_latex.split("\n")):
            if "\\being{document}" in row:
                start_index = index
                break
        new_latex = "\n".join(old_latex.split("\n")[start_index:])

        # colorize all non math text
        ignore_pos = self.get_ignore_positions(new_latex)
        for search_element in ["\\begin ", "\\end "]:
            new_latex = new_latex.replace(search_element, search_element[:-1])
        math_positions = {}
        for key in config_latex.color_patterns.keys():
            math_positions[key] = self.process_environment(new_latex, config_latex.color_patterns[key], str(key),
                                                  ignore_pos[key])
        return new_latex, math_positions

    def get_ignore_positions(self, new_latex):
        ignore_positions = {}
        for key in config_latex.ignore_patterns.keys():
            ignore_pos = {k: self.find_ignore_pos(new_latex, config_latex.ignore_patterns[key][k])
                          for k in config_latex.ignore_patterns[key].keys()}
            if ", " in key:
                key = key.split(", ")
            else:
                key = [key]
            for k in key:
                if k not in ignore_positions:
                    ignore_positions[k] = {}
                for kk in ignore_pos.keys():
                    ignore_positions[k][kk] = ignore_pos[kk]
        for key in ignore_positions:
            if 'bea' in ignore_positions[key]:
                for pos_i, pos in enumerate(ignore_positions[key]['bea']):
                    if new_latex[pos[0]:pos[1]].find('\\beq') > -1:
                        ignore_positions[key]['bea'][pos_i] = (
                        pos[0], pos[0] + new_latex[pos[0]:pos[1]].find('\\beq'))
        ignore_pos = {key: [] for key in config_latex.color_patterns.keys()}
        for key in ignore_positions.keys():
            for v in ignore_positions[key].values():
                ignore_pos[key] += v
        return ignore_pos

    def find_ignore_pos(self, data: str, search_pattern: str) -> Tuple[list, str]:
        """Process LaTeX environment."""
        vertices_ptrs = []
        start_pos = 0
        regex = re.compile(search_pattern)
        end_pattern = f"(?s)(?={search_pattern[search_pattern.find('.*?(?=') + 6:]}"
        if search_pattern == r"(?s)(?<=\$\$).*?(?=\$\$)":
            end_pattern == r"(?s)(?=\$\$)"
        while start_pos <= len(data):
            pattern = regex.search(data, pos=start_pos)
            # if found a pattern
            if pattern:
                # get start end position
                start, end = pattern.span()
                # check that no open brackets and $ exists
                while end_pattern:
                    open_brackets = len([match for match in re.finditer('{', data[start:end]) if
                                         data[start + match.span()[0] - 1] != '\\'])
                    closed_brackets = len([match for match in re.finditer('}', data[start:end]) if
                                           data[start + match.span()[0] - 1] != '\\'])
                    dollar_signs = len([match for match in re.finditer('\$', data[start:end]) if
                                        data[start + match.span()[0] - 1] != '\\'])
                    if open_brackets > closed_brackets or dollar_signs % 2 != 0:
                        new_end = re.search(end_pattern, data, pos=end + 1)
                        if new_end:
                            end = new_end.span()[0]
                        else:
                            break
                    else:
                        break
                if start == end:
                    start_pos = end + 1
                    continue
                found = data[start:end]
                if '\\begin{document}' in found:
                    start_pos = start + 1
                    continue
                vertices_ptrs.append((start, end))
                start_pos = end
            else:
                break
        return vertices_ptrs

if __name__ == '__main__':
    formula_file = "data/test.txt"
    output_file = "data/out.csv"
    categories = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "econ.EM", "econ.GN",
                  "econ.TH", "math.AC", "math.AG", "math.AP", "math.AT", "math.CO", "math.CV", "math.DG", "math.DS",
                  "math.LO", "astro-ph.CO", "astro-ph.GA", "astro-ph.HE", "hep-ph", "hep-th", "math-ph"]
    for category in categories:
        collect(output_file=formula_file, arxiv_args={'query': category, 'max_results': 500, 'sort_by': arxiv.SortCriterion.SubmittedDate},
                track_file="data/tracking_file_inline.csv").extract_formulas()
    tokenize(formula_file, output_file).process_file()
