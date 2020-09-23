import math

from numpy import ones, asarray
from numpy.random import default_rng

from torch import nn, Tensor, from_numpy

from joeynmt.helpers import freeze_params
from joeynmt.vocabulary import Vocabulary


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)


class PretrainedEmbeddings(Embeddings):

    """
    Loads embeddings from an embeddings file. Frozen by default.
    """
    def __init__(self,
                 embed_file: str,
                 vocab: Vocabulary,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = True,
                 **kwargs):
        super(PretrainedEmbeddings, self).__init__(embedding_dim, scale, vocab_size, padding_idx, freeze, **kwargs)

        # overwrite lut with embeddings from embed_file
        self.load_embeddings_from_file(embed_file, vocab)

        if freeze:
            freeze_params(self)

    def load_embeddings_from_file(self, embed_file: str, vocab: Vocabulary) -> None:
        """
        Overwrites the initial Embedding Tensor with the embeddings from the embedding file.
        Tokens without a specified embedding are initialised from a normal distribution.

        :param embed_file: path to file in word2vec text format
        :param vocab: the vocabulary of the model
        """
        # initialise all embeddings from a normal distribution
        loaded_embeds = default_rng().normal(size=(self.vocab_size, self.embedding_dim))

        # load embeddings from file if they are present in the vocabulary
        with open(embed_file, "r") as open_file:
            for line in open_file:
                line = line.strip()
                if line != "":
                    token, embedding_str = line.split(sep=" ", maxsplit=1)
                    idx = vocab.stoi.get(token, None)  # get index of token in vocabulary
                    if idx is None:  # token is not in vocabulary
                        continue
                    embedding = asarray(embedding_str.split(" "), dtype=float)
                    assert embedding.shape[0] == self.embedding_dim, "Dimensionality of loaded embedding does not match"
                    loaded_embeds[idx] = embedding  # replace with correct embedding

        # overwrite Embedding Tensor
        self.lut.weight.data.copy_(from_numpy(loaded_embeds))
