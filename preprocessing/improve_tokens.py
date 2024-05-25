import csv
import numpy as np

class improve_tokens():
    def __init__(self, files, correction_files={}, file_ending="_v2", use_only_c=False, remove=True):
        self.file_ending = file_ending
        self.replace_tokens_list = []
        self.remove_token_list = []
        self.text_style_elements = []
        self.before_bracket = []
        self.after_bracket = []
        self.two_brackets = []
        self.remove_formula = []
        self.remove = remove
        self.style_prob = 0.2
        with open("data/vocabs/token_types2.csv") as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                if not line[0]:
                    continue
                if line[1] == '3':
                    self.remove_token_list.append(line[0])
                elif line[2]:
                    self.replace_tokens_list.append([line[0].split(" "), line[2]])
                if line[1] == "1":
                    self.after_bracket.append(line[0])
                if line[1] == "2":
                    self.before_bracket.append(line[0])
                if line[1] == "5":
                    self.two_brackets.append(line[0])
                if line[1] == "6":
                    self.remove_formula.append(line[0])
                if line[3] == 'y':
                    self.text_style_elements.append(line[0])
        for file in files:
            self.file = file
            self.csv = True if file[-3:] == "csv" else False
            self.load_formulae(file)
            if self.file in correction_files.keys():
                for f in correction_files[self.file]:
                    self.load_manual_corrections(f)
            self.split_up_tokens()
            self.remove_tokens()
            self.split_up_non_latex_commands()
            self.replace_tokens()
            self.add_brackets()
            self.remove_empty_brackets()
            self.use_only_c = use_only_c
            self.process_array()
            if self.remove:
                self.remove_mathtype()
                self.remove_formula_if_token()
                self.first_token_check()
            self.remove_style_elements()
            self.remove_unnecessary_brackets()
            self.order_sub_sup()
            # self.add_styles()
            self.save_file(self.file)

    def load_formulae(self, file):
        """
        loading the formula from file
        Args:
            file: file with formulae

        Returns:

        """
        self.formulae = {}
        self.images = {}
        if self.csv:
            with open(file) as f:
                csv_reader = csv.reader(f)
                for line in csv_reader:
                    self.formulae[line[1]] = line[0].split(" ")
                    self.images[line[1]] = line[1:]
        else:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    image, formula = line.split(".png: ")
                    if " style: " in formula:
                        formula = formula.split(" style: ")[0]
                    self.formulae[image + ".png"] = formula.split(" ")
                    self.images[image + ".png"] = image + ".png"

    def save_file(self, file):
        """
        saving formulae in files
        Args:
            file: save file name

        Returns:

        """
        new_file = file[:-4] + self.file_ending
        if self.csv:
            with open(new_file, "w") as f:
                csv_writer = csv.writer(f)
                for image, formula in self.formulae.items():
                    csv_writer.writerow([" ".join(formula)] + self.images[image])
        else:
            with open(new_file, "w") as f:
                for image, formula in self.formulae.items():
                    if formula and formula[-1] == "\n":
                        formula.pop(-1)
                    formula_text = ' '.join(formula)
                    f.write(image + ': ' + formula_text + '\n')

    def load_manual_corrections(self, correction_file):
        """
        load manual correction files and update the corresponding formulae
        Args:
            correction_file: file with corrected formulae

        Returns:

        """
        with open(correction_file) as f:
            reader = csv.reader(f, delimiter=';')
            for line in reader:
                image, formula = line[0], line[1]
                if line[0] == "image":
                    continue
                if image + ".png" in self.formulae.keys():
                    self.formulae[image + ".png"] = formula.split(" ")
                    if formula == "DELETE":
                        self.formulae.pop(image + ".png")

    def replace_tokens(self):
        """
        replace tokens with similar tokens - reduce variation
        Returns:

        """
        for image, formula in self.formulae.items():
            new_formula_text = []
            wait_i = 0
            for token_i, token in enumerate(formula):
                if wait_i > 0:
                    wait_i -= 1
                    continue
                not_found = True
                for token_list in self.replace_tokens_list:
                    if formula[token_i:token_i+len(token_list[0])] == token_list[0]:
                        new_formula_text.append(token_list[1])
                        wait_i = len(token_list[0]) - 1
                        not_found = False
                        break
                if not_found:
                    new_formula_text.append(token)
            self.formulae[image] = new_formula_text

    def remove_tokens(self):
        """
        remove tokens from self.remove_token_list
        Returns:

        """
        for image, formula in self.formulae.items():
            self.formulae[image] = [token for token in formula if token not in self.remove_token_list]

    def remove_empty_brackets(self):
        """
        remove empty brackets { }
        Returns:

        """
        for image, formula in self.formulae.items():
            new_formula = []
            closing = False
            for token_i, token in enumerate(formula):
                if closing:
                    closing = False
                    continue
                if token != "{":
                    new_formula.append(token)
                    continue
                if token_i + 1 == len(formula):
                    continue
                if token == "{" and formula[token_i + 1] == "}":
                    closing = True
                    continue
                else:
                    new_formula.append(token)
            self.formulae[image] = new_formula

    def split_up_non_latex_commands(self):
        """
        if token contains \ not at the begining split up the token
        Returns:

        """
        for image, formula in self.formulae.items():
            new_formula_text = []
            for token in formula:
                if "\\" in token:
                    new_formula_text.append(token)
                else:
                    for element in token:
                        new_formula_text.append(element)
            self.formulae[image] = new_formula_text

    def process_array(self):
        """
        processing the array
        Returns:

        """
        delete_images = []
        for image, formula in self.formulae.items():
            new_formula = []
            new_formula_i = 0
            for token_i, token in enumerate(formula):
                # find \begin{array} token
                if token == "\\begin{array}":
                    if "\\end{array}" not in formula[token_i:]:
                        continue
                    end_array_i = token_i + formula[token_i:].index("\\end{array}")
                    not_good_defined_array = False
                    # get the array definition
                    if '}' not in formula[token_i:]:
                        continue
                    array_def_end_i = token_i + formula[token_i:].index("}")
                    array_def = []
                    for token in formula[token_i+1:array_def_end_i]:
                        if token == "" or token == "{" or token == "}":
                            continue
                        if self.use_only_c and token in ["l", "c", "r"]:
                            array_def.append("c")
                        else:
                            array_def.append(token)
                    # split up array into a 2d list
                    array_text = " ".join(formula[array_def_end_i+1:end_array_i])
                    array = []
                    # find array rows
                    for array_row in array_text.split("\\\\"):
                        row = []
                        if array_row == "":
                            continue
                        # find columns
                        for element in array_row.split("&"):
                            element = [e for e in element.split(" ") if e != ""]
                            if len(element) > 2 and element[0] == "{" and element[-1] == "}":
                                element.pop(0)
                                element.pop(-1)
                            # if element is missing
                            if not element:
                                not_good_defined_array = True
                            row.append(' '.join(element))
                        array.append(row)
                    # check that columns and definition of columns is correct
                    if not array or len(array_def) - sum([d == '|' for d in array_def]) != len(array[0]):
                        not_good_defined_array = True
                    # if only one row => remove array
                    if len(array) == 1:
                        new_formula += formula[new_formula_i:token_i] + " ".join(array[0]).split(" ")
                        new_formula_i = end_array_i + 1
                    else:
                        new_array_text = "{ " + ' '.join(array_def) + " } " + ' \\\\ '.join([' & '.join(array_row) for array_row in array])
                        new_formula += formula[new_formula_i:token_i + 1] + new_array_text.split(" ")
                        new_formula_i = end_array_i
                    # if array is not good defined delete formula
                    if not_good_defined_array and self.remove:
                        delete_images.append(image)
                        break
            if new_formula:
                self.formulae[image] = new_formula + formula[new_formula_i:]
        for image in delete_images:
            self.formulae.pop(image)

    def split_up_tokens(self):
        """
        split up following tokens
        Returns:

        """
        split_up_token_list = {
            "Object]": ["O", "b", "j", "e", "c", "t", "]"],
            "[object": ["[", "o", "b", "j", "e", "c", "t"],
            "\\left(": ["\\left", "("],
            "\\left.": ["\\left", "."],
            "\\left<": ["\\left", "<"],
            "\\left[": ["\\left", "["],
            "\\left\\langle": ["\\left", "\\langle"],
            "\\left\\lbrace": ["\\left", "\\lbrace"],
            "\\left\\lbrack": ["\\left", "\\lbrack"],
            "\\left\\lfloor": ["\\left", "\\lfloor"],
            "\\left\\vert": ["\\left", "\\vert"],
            "\\left\\{": ["\\left", "\\{"],
            "\\left\\|": ["\\left", "\\|"],
            "\\left|": ["\\left", "|"],
            "\\left/": ["\\left", "/"],
            "\\left\\lceil": ["\\left", "\\lceil"],
            "\\left]": ["\\left", "]"],
            "\\right)": ["\\right", ")"],
            "\\right.": ["\\right", "."],
            "\\right>": ["\\right", ">"],
            "\\right\\rangle": ["\\right", "\\rangle"],
            "\\right\\rbrace": ["\\right", "\\rbrace"],
            "\\right\\rbrack": ["\\right", "\\rbrack"],
            "\\right\\rfloor": ["\\right", "\\rfloor"],
            "\\right\\vert": ["\\right", "\\vert"],
            "\\right\\|": ["\\right", "\\|"],
            "\\right\\}": ["\\right", "\\}"],
            "\\right]": ["\\right", "]"],
            "\\right|": ["\\right", "|"],
            "\\right/": ["\\right", "/"],
            "\\right[": ["\\right", "["],
            "\\right\\rceil": ["\\right", "\\rceil"],
            ",\\;": [",", "\\;"],
            ",\\qquad": [",", "\\qquad"],
            "\\rm\\bf": ["\\rm", "\\bf"],
            "\\scriptstyle\\lambda": ["\\scriptstyle", "\\lambda"],
            "\\mu\\nu": ["\\mu", "\\nu"],
            "\\mu\\alpha": ["\\mu", "\\alpha"],
            "\\gtM": ["\\gt", "M"],
            "\\gtp": ["\\gt", "p"],
            "\\ltN": ["\\lt", "N"],
            "\\ltl": ["\\lt", "l"],
            "\\ltq": ["\\lt", "q"]
        }
        for image, formula in self.formulae.items():
            new_formula_text = []
            for token in formula:
                if token in split_up_token_list.keys():
                    new_formula_text += split_up_token_list[token]
                else:
                    new_formula_text.append(token)
            self.formulae[image] = new_formula_text

    def remove_mathtype(self):
        """
        remove formulae that contain MathType!
        Returns:

        """
        delete_images = []
        for image, formula in self.formulae.items():
            formula_text = " ".join(formula)
            if "M a t h T y p e !" in formula_text:
                delete_images.append(image)
        for image in delete_images:
            self.formulae.pop(image)

    def first_token_check(self):
        """
        delete formulae that start with _ or ^
        Returns:

        """
        delete_images = []
        for image, formula in self.formulae.items():
            if len(formula) == 0:
                continue
            if formula[0] in ['_', '^']:
                delete_images.append(image)
        for image in delete_images:
            self.formulae.pop(image)

    def add_brackets(self):
        """
        add brackets for frac
        Returns:

        """
        for image, formula in self.formulae.items():
            corr_i = 0
            for i in [idx for idx, value in enumerate(formula) if value == "\\frac"]:
                if i + 1 + corr_i >= len(formula):
                    formula.pop()
                    break
                if formula[i + 1 + corr_i] != "{":
                    formula.insert(i + 1 + corr_i, "{")
                    formula.insert(i + 3 + corr_i, "}")
                    formula.insert(i + 4 + corr_i, "{")
                    formula.insert(i + 6 + corr_i, "}")
                    corr_i += 4
            corr_i = 0
            for i in [idx for idx, value in enumerate(formula) if (value == "_" or value == "^")]:
                if i + 1 + corr_i >= len(formula):
                    formula.pop()
                    break
                if formula[i + 1 + corr_i] != "{":
                    formula.insert(i + 1 + corr_i, "{")
                    formula.insert(i + 3 + corr_i, "}")
                    corr_i += 2
            self.formulae[image] = formula

    def order_sub_sup(self):
        """
        bring all sub and sup in the same order => first sup and second sub
        Returns:

        """
        for image, formula in self.formulae.items():
            new_formula = []
            end_i = 0
            for i in [idx for idx, value in enumerate(formula) if (value == "^" or value == "_")]:
                if i < end_i:
                    continue
                sup = []
                sub = []
                new_formula += formula[end_i:i]
                end_i = i
                while formula[end_i] == "_" or formula[end_i] == "^":
                    if end_i+1 >= len(formula):
                        end_i += 1
                        break
                    if formula[end_i+1] == "_" or formula[end_i+1] == "^":
                        end_i += 1
                        continue
                    if formula[end_i] == "_":
                        closing_bracket = self.find_closing_bracket(formula[end_i+1:])
                        sub += formula[end_i + 2:end_i + closing_bracket]
                        end_i = end_i + 1 + closing_bracket
                    elif formula[end_i] == "^":
                        closing_bracket = self.find_closing_bracket(formula[end_i+1:])
                        sup += formula[end_i + 2:end_i + closing_bracket]
                        end_i = end_i + 1 + closing_bracket
                    if end_i >= len(formula):
                        break
                if sub:
                    new_formula += ["_", "{"] + sub + ["}"]
                if sup:
                    new_formula += ["^", "{"] + sup + ["}"]
            new_formula += formula[end_i:]
            self.formulae[image] = new_formula

    @staticmethod
    def find_closing_bracket(formula):
        """
        finding the corresponding closing bracket
        Args:
            formula: rest formula from the opening bracket

        Returns: closing bracket index

        """
        open_brackets = 0
        closed_brackets = 0
        i = 0
        for i, token in enumerate(formula):
            if token == "{": open_brackets += 1
            if token == "}": closed_brackets += 1
            if open_brackets == closed_brackets:
                break
        return i + 1

    def remove_style_elements(self):
        """
        remove style tokens
        Returns:

        """
        for image, formula in self.formulae.items():
            self.formulae[image] = [token for token in formula if token not in self.text_style_elements]

    def remove_unnecessary_brackets(self):
        """
        remove unnecessary brackets from formulae
        Returns:

        """
        for image, formula in self.formulae.items():
            if "{" not in formula:
                continue
            second_bracket = -1
            drop_brackets = []
            for token_i, token in enumerate(formula):
                if token_i == second_bracket:
                    continue
                if token == "{":
                    if token_i > 0 and formula[token_i - 1] in self.before_bracket:
                        continue
                    if token_i + 1 < len(formula):
                        open_brackets = 0
                        closed_brackets = 0
                        found = False
                        for token in formula[token_i + 1:token_i + self.find_closing_bracket(formula[token_i:])]:
                            if token == '{':
                                open_brackets += 1
                            if token == '}':
                                closed_brackets += 1
                            if token in self.after_bracket and open_brackets == closed_brackets:
                                found = True
                                break
                        if found:
                            continue
                    if token_i > 0 and formula[token_i - 1] in self.two_brackets:
                        second_bracket = token_i + self.find_closing_bracket(formula[token_i:])
                        continue
                    drop_brackets.append(token_i)
                    drop_brackets.append(token_i + self.find_closing_bracket(formula[token_i:]) - 1)
            self.formulae[image] = [token for token_i, token in enumerate(formula) if token_i not in drop_brackets]

    def remove_formula_if_token(self):
        """
        remove complete formula if token in formula
        Returns:

        """
        pop_images = []
        for image, formula in self.formulae.items():
            if any(token in self.remove_formula for token in formula):
                pop_images.append(image)
        for pop_image in pop_images:
            self.formulae.pop(pop_image)

    def add_styles(self):
        bolt_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\\alpha', '\\beta', '\\gamma',
                     '\\delta', '\\epsilon', '\\varepsilon', '\\zeta', '\\eta', '\\theta', '\\vartheta', '\\Theta',
                     '\\iota', '\\kappa', '\\lambda', '\\Lambda', '\\mu', '\\nu', '\\xi', '\\Xi', '\\pi', '\\Pi', '\\rho',
                     '\\varrho', '\\sigma', '\\Sigma', '\\tau', '\\phi', '\\varphi', '\\Phi', '\\chi', '\\psi', '\\Psi',
                     '\\omega', '\\Omega']
        cal_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
        mathbb_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z', '1']
        new_formulae = {}
        for image, formula in self.formulae.items():
            i = 0
            while True:
                i += 1
                new_formula = []
                for token in formula:
                    change_prob = np.random.rand(1)
                    if change_prob < self.style_prob/3 and token in bolt_list:
                        # do bolt
                        if len(new_formula) > 1 and new_formula[-2] in ['\\dot', '\\ddot', '\\hat', '\\bar', '\\vec', '\\tilde']:
                            temp = new_formula[-2:]
                            new_formula = new_formula[:-2] + ['\\boldsymbol', '{'] + temp + [token, '}']
                        else:
                            new_formula += ['\\boldsymbol', '{', token, '}']
                    elif self.style_prob/3 < change_prob < 2 * self.style_prob/3 and token in cal_list:
                        new_formula.append('\\mathcal{' + token + '}')
                    elif 2 * self.style_prob/3 < change_prob < self.style_prob and token in mathbb_list:
                        new_formula.append('\\mathbb{' + token + '}')
                    else:
                        new_formula.append(token)
                if formula != new_formula or i == 10:
                    new_formulae[image] = new_formula
                    break
        self.formulae = new_formulae

if __name__ == '__main__':
    improve_tokens(files=["your files"], correction_files={},
                   file_ending="_normalized.csv", use_only_c=True)