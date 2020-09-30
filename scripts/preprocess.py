#!/usr/bin/env python3
"""
Normalizes and tokenizes every sentence in a corpus using the Moses normalizer and tokenizer.

Takes three arguments:
* the language of the corpus (language code)
* the path to the corpus file
* the path to the output file
"""
from sacremoses import MosesPunctNormalizer, MosesTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lang", type=str, help="Language of the corpus")
parser.add_argument("f_in", type=str, help="Path to the corpus")
parser.add_argument("f_out", type=str, help="Output path")

args = parser.parse_args()

normalizer = MosesPunctNormalizer(args.lang)
tokenizer = MosesTokenizer(args.lang)

with open(args.f_in, 'r', encoding='UTF-8') as f_in, open(args.f_out, 'w', encoding='UTF-8') as f_out:
    for line in f_in:
        line = line.strip()
        if line != '':
            line = normalizer.normalize(line)
            line = tokenizer.tokenize(line, return_str=True, escape=False)
            f_out.write(line + '\n')
