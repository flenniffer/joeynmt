# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep

    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


def load_unsupervised_data(data_cfg: dict) \
        -> (Dataset, Dataset, Dataset, Dataset,
            Dataset, Dataset,
            Optional[Dataset], Optional[Dataset],
            Vocabulary, Vocabulary):
    """
    :param data_cfg: configuration dictionary for data
    :return:
        - src2src:
        - trg2trg:
        - BTsrc:
        - BTtrg:
        - dev_src2trg:
        - dev_trg2src:
        - test_src2trg:
        - test_trg2src:
        - src_vocab:
        - trg_vocab:
    """
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    noised_ext = data_cfg["noised"]
    denoised_ext = data_cfg["denoised"]
    assert noised_ext != denoised_ext
    train_path = data_cfg["train"]
    src2trg_dev_path = data_cfg["src2trg_dev"]
    trg2src_dev_path = data_cfg["trg2src_dev"]
    src2trg_test_path = data_cfg.get("src2trg_test", None)
    trg2src_test_path = data_cfg.get("trg2src_test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    # datasets for denoising
    # 'translate' from noised input to denoised output
    src2src = TranslationDataset(path=train_path,
                                 exts=("." + noised_ext + "." + src_lang,
                                       "." + denoised_ext + "." + src_lang),
                                 fields=(src_field, trg_field),
                                 filter_pred=
                                 lambda x: len(vars(x)['src']) <= max_sent_length
                                 and len(vars(x)['trg']) <= max_sent_length)

    trg2trg = TranslationDataset(path=train_path,
                                 exts=("." + noised_ext + "." + trg_lang,
                                       "." + denoised_ext + "." + trg_lang),
                                 fields=(src_field, trg_field),
                                 filter_pred=
                                 lambda x: len(vars(x)['src']) <= max_sent_length
                                 and len(vars(x)['trg']) <= max_sent_length)

    # datasets for BT
    # need denoised src data in order to create back-translation on-the-fly
    # then use (BT, denoised trg data) tuples as training examples
    # so for now, create (denoised src data, denoised trg data) datasets in the same language
    BTsrc = TranslationDataset(path=train_path,
                               exts=("." + denoised_ext + "." + src_lang,
                                     "." + denoised_ext + "." + src_lang),
                               fields=(src_field, trg_field),
                               filter_pred=
                               lambda x: len(vars(x)['src']) <= max_sent_length
                               and len(vars(x)['trg']) <= max_sent_length)

    BTtrg = TranslationDataset(path=train_path,
                               exts=("." + denoised_ext + "." + trg_lang,
                                     "." + denoised_ext + "." + trg_lang),
                               fields=(src_field, trg_field),
                               filter_pred=
                               lambda x: len(vars(x)['src']) <= max_sent_length
                               and len(vars(x)['trg']) <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="trg", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=src2src, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=trg2trg, vocab_file=trg_vocab_file)

    def _choose_random_subset(train: Dataset, train_subset: int):
        keep_ratio = train_subset / len(train)
        keep, _ = train.split(split_ratio=[keep_ratio, 1 - keep_ratio],
                              random_state=random.getstate())
        return keep

    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        src2src = _choose_random_subset(src2src, random_train_subset)
        trg2trg = _choose_random_subset(trg2trg, random_train_subset)
        BTsrc = _choose_random_subset(BTsrc, random_train_subset)
        BTtrg = _choose_random_subset(BTtrg, random_train_subset)

    assert len(src2src) == len(trg2trg) == len(BTsrc) == len(BTtrg), \
        "All training sets must have equal length for unsupervised NMT."

    dev_src2trg = TranslationDataset(path=src2trg_dev_path,
                                     exts=("." + src_lang, "." + trg_lang),
                                     fields=(src_field, trg_field))

    dev_trg2src = TranslationDataset(path=trg2src_dev_path,
                                     exts=("." + trg_lang, "." + src_lang),
                                     fields=(src_field, trg_field))

    def _make_test_set(test_path: str, src_lang: str, trg_lang: str) -> Optional[Dataset]:
        if test_path is not None:
            if os.path.isfile(test_path + "." + trg_lang):
                return TranslationDataset(path=test_path,
                                          exts=("." + src_lang, "." + trg_lang),
                                          fields=(src_field, trg_field))
            else:
                return MonoDataset(path=test_path,
                                   ext="." + src_lang,
                                   field=src_field)
        else:
            return None

    test_src2trg = _make_test_set(src2trg_test_path,
                                  src_lang=src_lang, trg_lang=trg_lang)
    test_trg2src = _make_test_set(trg2src_test_path,
                                  src_lang=trg_lang, trg_lang=src_lang)

    return src2src, trg2trg, BTsrc, BTtrg, dev_src2trg, dev_trg2src, test_src2trg, test_trg2src, src_vocab, trg_vocab

# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)
