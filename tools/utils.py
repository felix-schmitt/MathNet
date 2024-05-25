import subprocess
from threading import Timer
import torch
import yaml
import numpy as np
import nltk
import tabulate
import Levenshtein

def run(cmd, timeout_sec):
    proc = subprocess.Popen(cmd, shell=True)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def load_config(config_file, arguments=None):
    with open(config_file) as f:
        config = yaml.load(f, yaml.loader.SafeLoader)

    if torch.cuda.is_available():
        device0 = 'cuda'
    elif torch.backends.mps.is_available():
        device0 = "mps"
    else:
        device0 = 'cpu'

    print("devices: ")
    if device0 == 'cuda':
        device_data = [[device0, torch.cuda.get_device_name(device0), torch.cuda.get_device_properties(device0)]]
        print(tabulate.tabulate(device_data, headers=["id", "type", "properties"], tablefmt="fancy_outline"))
    else:
        print(device0)
    config['device'] = device0

    if arguments:
        config['arguments'] = vars(arguments)
        if config['arguments']['turn_off_wandb']:
            config['wandb']['use'] = False
        if config['arguments']['num_workers']:
            config['dataset']['train']['num_workers'] = config['arguments']['num_workers']
            config['dataset']['val']['num_workers'] = config['arguments']['num_workers']
            if 'test' in config['dataset']:
                config['dataset']['test']['num_workers'] = config['arguments']['num_workers']
        if config['arguments']['test_set']:
            config['dataset']['test']['files'] = {"test": [config['arguments']['test_set']]}
    if 'train' in config['dataset']:
        config['train']['init_lr'] *= config['dataset']['train']['batch_size']/48
    return config

def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for line in f:
            if ".png: " not in line:
                continue
            image, formula = line.split(".png: ")
            formula = formula[:-1]
            if " style: " in formula:
                formula = formula.split(" style: ")[0]
            if image[-4:] != ".png":
                formulas[image + ".png"] = formula.split(' ')
            else:
                formulas[image] = formula.split(' ')

    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas


def score_files(path_ref, path_hyp, more_information=False):
    """Loads result from file and score it

    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.
        more_information: (boolean) if should return more information

    Returns:
        scores: (dict)

    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # score
    if not more_information:
        return {
            "BLEU-4": bleu_score(formulas_ref, formulas_hyp) * 100,
            "EM": exact_match_score(formulas_ref, formulas_hyp) * 100,
            "Edit": edit_distance(formulas_ref, formulas_hyp) * 100
        }
    else:
        edit, ops, errors, n_errors = edit_distance_extended(formulas_ref, formulas_hyp)
        return {
            "BLEU-4": bleu_score(formulas_ref, formulas_hyp) * 100,
            "EM": exact_match_score(formulas_ref, formulas_hyp) * 100,
            "Edit":  edit * 100,
            "ops": ops,
            "errors": errors,
            **n_errors
        }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect

    """
    exact_match = 0
    for image, hypo in hypotheses.items():
        ref = references[image]
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.

    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)

    Returns:
        BLEU-4 score: (float)

    """
    references = [[references[image]] for image in hypotheses.keys()]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses.values(), weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of token (one hypothesis)
        hypotheses: list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for image, hypo in hypotheses.items():
        ref = references[image]
        d_leven += Levenshtein.distance(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


def edit_distance_extended(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of token (one hypothesis)
        hypotheses: list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    errors = {}
    ops = {'insert': 0, 'replace': 0, 'delete': 0}
    n_errors = {"0": 0, "1": 0, "3": 0, "5": 0, "10": 0, "30": 0, "50": 0}
    with_array_d, without_array_d, tiny_d, small_d, middle_d, large_d = 0, 0, 0, 0, 0, 0
    with_array_len, without_array_len, tiny_len, small_len, middle_len, large_len = 0, 0, 0, 0, 0, 0
    for image, hypo in hypotheses.items():
        ref = references[image]
        operations = Levenshtein.editops(hypo, ref)
        for operation in operations:
            ops[operation[0]] += 1
            if operation[0] == 'delete':
                token = hypo[operation[1]]
            else:
                token = ref[operation[2]]
            if token in errors:
                errors[token] += 1
            else:
                errors[token] = 1
        d = Levenshtein.distance(ref, hypo)
        for key in n_errors.keys():
            if d <= int(key):
                n_errors[key] += 1
        if "\\begin{array}" in ref:
            with_array_d += d
            with_array_len += float(max(len(ref), len(hypo)))
        else:
            without_array_d += d
            without_array_len += float(max(len(ref), len(hypo)))
        if len(ref) < 10:
            tiny_d += d
            tiny_len += float(max(len(ref), len(hypo)))
        elif len(ref) < 50:
            small_d += d
            small_len += float(max(len(ref), len(hypo)))
        elif len(ref) < 100:
            middle_d += d
            middle_len += float(max(len(ref), len(hypo)))
        else:
            large_d += d
            large_len += float(max(len(ref), len(hypo)))
        d_leven += d
        len_tot += float(max(len(ref), len(hypo)))
    errors = dict(sorted(errors.items(), key=lambda item: item[1], reverse=True))
    for key, value in ops.items():
        ops[key] = (1 - ops[key] / len_tot) * 100
    total_errors = sum([error for error in errors.values()])
    for key, value in errors.items():
        errors[key] = (errors[key] / total_errors) * 100 if total_errors > 0 else 0
    for key, value in n_errors.items():
        n_errors[key] = (n_errors[key] / len(hypotheses)) * 100 if len(hypotheses) > 0 else 0
    n_errors["with_array"] = 100 - with_array_d/with_array_len * 100 if with_array_len > 0 else 0
    n_errors["without_array"] = 100 - without_array_d/without_array_len * 100 if without_array_len > 0 else 0
    n_errors["tiny"] = 100 - tiny_d / tiny_len * 100 if tiny_len > 0 else 0
    n_errors["small"] = 100 - small_d / small_len * 100 if small_len > 0 else 0
    n_errors["middle"] = 100 - middle_d / middle_len * 100 if middle_len > 0 else 0
    n_errors["large"] = 100 - large_d / large_len * 100 if large_len > 0 else 0
    return 1. - d_leven / len_tot, ops, errors, n_errors


def get_accuracy(labels, predictions, end_number):
    """calculating accuracy with dropping empty formula"""
    accuracy = []
    for batch_i in range(len(labels)):
        i_end_gt = (labels[batch_i] == end_number).nonzero(as_tuple=True)[0] - 1
        i_end_pre = (predictions[batch_i] == end_number).nonzero(as_tuple=True)[0]
        if predictions[batch_i].shape != torch.Size([]):
            prediction_len = torch.tensor([len(predictions[batch_i])]).to(labels.device)
        else:
            prediction_len = torch.tensor([0]).to(labels.device)
        i_max = torch.max(i_end_gt[0] if len(i_end_gt) > 0 else torch.tensor([len(labels[batch_i]) - 1]).to(labels.device),
                          i_end_pre[0] if len(i_end_pre) > 0 else torch.tensor([len(predictions[batch_i])]).to(labels.device))
        i_end = torch.min(i_max, prediction_len)
        acc = torch.true_divide(torch.sum(predictions[batch_i][:i_end] == labels[batch_i][1:i_end+1]), i_max)
        if acc.shape != torch.Size([]):
            accuracy.append(acc)
        else:
            accuracy.append(torch.tensor([acc]).to(labels.device))
    return torch.mean(torch.stack(accuracy)) * 100

def remove_style(predictions):
    for i, prediction in enumerate(predictions):
        new_prediction = []
        wait = 0
        for token_i, token in enumerate(prediction):
            if wait > 0:
                wait -= 1
                continue
            if token == "\\boldsymbol":
                close_bracket = find_closing_bracket(prediction[token_i+1:])
                new_prediction += prediction[token_i+1:token_i+1+close_bracket]
                wait = close_bracket
                continue
            if "\\mathcal" in token:
                new_prediction.append(token.replace("\\mathcall{", "").replace("}", ""))
                continue
            if "\\mathbb" in token:
                new_prediction.append(token.replace("\\mathbb{", "").replace("}", ""))
                continue
            new_prediction.append(token)
        predictions[i] = new_prediction
    return predictions


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


def gen_counting_label(labels, channel, ignore):
    b, t = labels.size()
    device = labels.device
    counting_labels = torch.zeros((b, channel))
    for i in range(b):
        for j in range(t):
            k = labels[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1
    return counting_labels.to(device)