#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import argparse
from typing import List, Set
from collections import Counter


def load_vocab(file: str) -> Set[str]:
    """
    Loads the vocabulary into a set.

    :param file: Path to vocabulary file
    :return: set of tokens in the vocabulary
    """
    vocab = set()
    with open(file, encoding="utf-8") as f_in:
        for token in f_in:
            token = token.strip()
            if token == "":
                continue
            vocab.add(token)

    return vocab


def get_coverage_dict(vocab: Set[str], data_files: List[str], names: List[str], max_unks: int = 3) -> dict:
    """
    Returns a dictionary to plot the vocabulary coverage.

    :param vocab: Vocabulary
    :param data_files: Files to analyse
    :param names: Names of the data splits
    :param max_unks: Maximum number of unknown tokens to count
    :return: Coverage dictionary
    """
    coverage_dict = {idx: Counter() for idx in range(len(names))}

    for file, idx in zip(data_files, range(len(names))):
        with open(file, encoding="utf-8") as f:
            for sent in f:
                sent = sent.strip()
                if sent == "":
                    continue
                sent_unk = 0
                if args.lower_data:
                    sent = sent.lower()
                for token in sent.split(" "):
                    if token not in vocab:
                        sent_unk += 1
                        if sent_unk == max_unks:
                            break
                coverage_dict[idx][sent_unk] += 1
                coverage_dict[idx]["total"] += 1

    for idx, unk_counter in coverage_dict.items():
        for unk_count in range(max_unks+1):
            unk_counter[unk_count] /= unk_counter["total"]
        del unk_counter["total"]

    return coverage_dict


def plot_coverage(src_coverage: dict, trg_coverage: dict,
                  names: List[str], max_unks: int = 3,
                  output_path: str = "vocab.png") -> None:
    """
    Plots the vocabulary coverage using the source and target language coverage dictionaries.

    :param src_coverage: Source language coverage dict
    :param trg_coverage: Target language coverage dict
    :param names: Names of the data splits
    :param max_unks: Maximum number of unks counted
    :param output_path: Filepath to save the plot to
    """
    f, axes = plt.subplots(len(names), sharex="col", sharey="col", figsize=(max_unks/1.5, 3*len(names)), squeeze=False)

    unk_counts = list(range(max_unks + 1))

    for idx, counter in src_coverage.items():
        unk_percentage = [counter[unk_count] for unk_count in unk_counts]
        axes[idx][0].scatter(unk_counts, unk_percentage, label=args.src_lang)

    for idx, counter in trg_coverage.items():
        unk_percentage = [counter[unk_count] for unk_count in unk_counts]
        axes[idx][0].scatter(unk_counts, unk_percentage, label=args.trg_lang, marker="^")

    for row in range(len(names)):
        axes[row][0].set_ylabel(names[row])
    axes[-1][0].set_xlabel("Number of unknown tokens per sentence")
    axes[-1][0].set_xticks([float(x) for x in range(max_unks + 1)])
    axes[-1][0].set_xticklabels([str(x) for x in range(max_unks)] + [f"{max_unks}+"])
    axes[0][0].set_title("Vocabulary coverage")
    axes[0][0].legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot vocabulary coverage.")
    parser.add_argument("--src_vocab", type=str, help="Source language vocabulary file")
    parser.add_argument("--trg_vocab", type=str, help="Target language vocabulary file")
    parser.add_argument("--src_data", type=str, nargs="+", help="Source language data files")
    parser.add_argument("--trg_data", type=str, nargs="+", help="Target language data files")
    parser.add_argument("--data_splits", type=str, nargs="+", help="Splits of given data, for legend")
    parser.add_argument("--max_unks", type=int, default=3, help="Maximum unknowns to count in a sentence")
    parser.add_argument("--output_path", type=str, default="vocab.png", help="Plot will be saved to this file")
    parser.add_argument("--lower_data", type=bool, default=False,
                        help="Set if data should be lowercased before counting")
    parser.add_argument("src_lang", type=str, help="Source language, for legend")
    parser.add_argument("trg_lang", type=str, help="Target language, for legend")

    args = parser.parse_args()
    assert len(args.src_data) == len(args.trg_data) == len(args.data_splits)
    src_vocab = load_vocab(args.src_vocab)
    trg_vocab = load_vocab(args.trg_vocab)

    src_coverage = get_coverage_dict(src_vocab, args.src_data, args.data_splits, args.max_unks)
    trg_coverage = get_coverage_dict(trg_vocab, args.trg_data, args.data_splits, args.max_unks)

    plot_coverage(src_coverage, trg_coverage, args.data_splits, args.max_unks, args.output_path)
