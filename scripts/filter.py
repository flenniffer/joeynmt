#!/usr/bin/env python3
"""
Filters segments from a corpus if they are shorter or longer than a given threshold.
Length is measured as number of elements after splitting at whitespace.

Takes four arguments:
* path to the corpus file to be filtered
* path to the output file
* minimum number of elements per segment in the output file
* maximum number of elements per segment in the output file
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("f_in", type=str, help="Path to the corpus file")
parser.add_argument("f_out", type=str, help="Path to the output file")
parser.add_argument("--min", type=int, default=5, help="Minimum number of elements per sentence")
parser.add_argument("--max", type=int, default=30, help="Maximum number of elements per sentence")

args = parser.parse_args()
min_threshold = args.min
max_threshold = args.max

with open(args.f_in) as f_in, open(args.f_out, 'w', encoding='UTF-8') as f_out:
    for segment in f_in:
        tokens = len(segment.split())
        if tokens < min_threshold or tokens > max_threshold:
            continue
        f_out.write(segment)
