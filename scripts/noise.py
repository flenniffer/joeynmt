#!/usr/bin/env python3
"""
Creates a noised version of a given corpus of sentences.

Noise is introduced in the form of swapping some words with neighboring words randomly.
The percentage of words swapped within every sentence is set as a parameter.

Takes three arguments:
* path to the corpus file to be noised
* path to the output file
* swapping percentage
"""
import random
from math import ceil
import argparse


def noise_sentence(sentence: str, percentage: float) -> str:
    """
    Chooses a percentage of tokens in the sentence and randomly swaps them with a neighbouring token.

    If the first (last) token is to be swapped with its left (right) neighbour, no swap is made.
    :param sentence: Sentence to noise
    :param percentage: Percentage of tokens to swap within the sentence
    :return: Noised sentence
    """
    split_sentence = sentence.split(" ")
    tokens = len(split_sentence)
    idxs = range(tokens)
    amount_to_swap = ceil(tokens * percentage)
    idx_to_swap = random.sample(idxs, k=amount_to_swap)
    noised_idxs = list(idxs)
    for idx in idx_to_swap:
        swap_left = bool(random.getrandbits(1))
        if swap_left and idx > 0:
            noised_idxs[idx-1], noised_idxs[idx] = noised_idxs[idx], noised_idxs[idx-1]
        elif not swap_left and idx < tokens-1:
            noised_idxs[idx], noised_idxs[idx+1] = noised_idxs[idx+1], noised_idxs[idx]

    return " ".join([split_sentence[idx] for idx in noised_idxs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("f_in", type=str, help="Path to the corpus")
    parser.add_argument("f_out", type=str, help="Path to the output file")
    parser.add_argument("swapping_perc", type=float, help="Percentage of words swapped in every sentence")

    args = parser.parse_args()

    with open(args.f_in, 'r', encoding='UTF-8') as f_in, open(args.f_out, 'w', encoding='UTF-8') as f_out:
        for line in f_in:
            line = line.strip()
            if line != "":
                noised = noise_sentence(line, args.swapping_perc)
                f_out.write(noised + "\n")
