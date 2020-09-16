# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings, PretrainedEmbeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch
from joeynmt.helpers import ConfigurationError


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable
        out, hidden, att_probs, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask)

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # greedy decoding
        if beam_size < 2:
            stacked_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output, eos_index=self.eos_index,
                    src_mask=batch.src_mask, embed=self.trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)


class UnsupervisedNMTModel:
    def __init__(self,
                 loaded_src_embed: PretrainedEmbeddings,
                 loaded_trg_embed: PretrainedEmbeddings,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 encoder: Encoder,
                 src_decoder: Decoder,
                 trg_decoder: Decoder,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary
                 ) -> None:
        """
        Create a new unsupervised NMT model.
        Architecture follows Artetxe et al. (2018): Unsupervised Neural Machine Translation.
        TODO
        :param src_embed:
        :param trg_embed:
        :param encoder:
        :param src_decoder:
        :param trg_decoder:
        :param src_vocab:
        :param trg_vocab:
        """
        super(UnsupervisedNMTModel, self).__init__()
        self.loaded_src_embed = loaded_src_embed
        self.loaded_trg_embed = loaded_trg_embed
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.shared_encoder = encoder
        self.src_decoder = src_decoder
        self.trg_decoder = trg_decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.src_pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.src_eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.trg_bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.trg_pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.trg_eos_index = self.trg_vocab.stoi[EOS_TOKEN]

        # Need four translators for four directions
        # To optimize individual parameters
        self.src2src_translator = Model(self.shared_encoder, self.src_decoder,
                                        self.loaded_src_embed, self.src_embed,
                                        self.src_vocab, self.src_vocab)
        self.src2trg_translator = Model(self.shared_encoder, self.trg_decoder,
                                        self.loaded_src_embed, self.trg_embed,
                                        self.src_vocab, self.trg_vocab)
        self.trg2src_translator = Model(self.shared_encoder, self.src_decoder,
                                        self.loaded_trg_embed, self.src_embed,
                                        self.trg_vocab, self.src_vocab)
        self.trg2trg_translator = Model(self.shared_encoder, self.trg_decoder,
                                        self.loaded_trg_embed, self.trg_embed,
                                        self.trg_vocab, self.trg_vocab)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return f"{self.__class__.__name__}(\n" \
               f"\tencoder={self.shared_encoder},\n" \
               f"\tsrc_decoder={self.src_decoder},\n" \
               f"\ttrg_decoder={self.trg_decoder},\n" \
               f"\tsrc_embed={self.src_embed},\n" \
               f"\ttrg_embed={self.trg_embed})"


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    if cfg.get("architecture", "encoder-decoder") == "encoder-decoder":
        model = build_encoder_decoder_model(cfg, src_vocab, trg_vocab)
    elif cfg.get("architecture", "encoder-decoder") == "unsupervised-nmt":
        model = build_unsupervised_nmt_model(cfg, src_vocab, trg_vocab)
    else:
        raise ConfigurationError("Supported architectures: 'encoder-decoder' and 'unsupervised-nmt'")

    return model


def build_encoder_decoder_model(cfg: dict = None,
                                src_vocab: Vocabulary = None,
                                trg_vocab: Vocabulary = None) -> Model:

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model


def build_unsupervised_nmt_model(cfg: dict = None,
                                 src_vocab: Vocabulary = None,
                                 trg_vocab: Vocabulary = None) -> UnsupervisedNMTModel:
    """
    TODO
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    # build source and target embedding layers
    loaded_src_embed = PretrainedEmbeddings(**cfg["encoder"]["embeddings"],
                                            vocab_size=len(src_vocab),
                                            padding_idx=src_padding_idx,
                                            vocab=src_vocab)

    loaded_trg_embed = PretrainedEmbeddings(**cfg["decoder"]["embeddings"],
                                            vocab_size=len(trg_vocab),
                                            padding_idx=trg_padding_idx,
                                            vocab=trg_vocab,
                                            freeze=True)

    src_embed = Embeddings(**cfg["encoder"]["embeddings"],
                           vocab_size=len(src_vocab),
                           padding_idx=src_padding_idx,
                           freeze=False)

    trg_embed = Embeddings(**cfg["decoder"]["embeddings"],
                           vocab_size=len(trg_vocab),
                           padding_idx=trg_padding_idx,
                           freeze=False)

    # build shared encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        shared_encoder = TransformerEncoder(**cfg["encoder"],
                                            emb_size=src_embed.embedding_dim,
                                            emb_dropout=enc_emb_dropout)
    else:
        shared_encoder = RecurrentEncoder(**cfg["encoder"],
                                          emb_size=src_embed.embedding_dim,
                                          emb_dropout=enc_emb_dropout)

    # build source and target decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        src_decoder = TransformerDecoder(
            **cfg["decoder"], encoder=shared_encoder, vocab_size=len(src_vocab),
            emb_size=src_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        trg_decoder = TransformerDecoder(
            **cfg["decoder"], encoder=shared_encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        src_decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=shared_encoder, vocab_size=len(src_vocab),
            emb_size=src_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        trg_decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=shared_encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = UnsupervisedNMTModel(loaded_src_embed, loaded_trg_embed,
                                 src_embed, trg_embed,
                                 shared_encoder,
                                 src_decoder, trg_decoder,
                                 src_vocab, trg_vocab)

    # initialise model, embed_initializer should be none
    # so loaded embeddings won't be overwritten
    initialize_model(model.src2src_translator, cfg, src_padding_idx, src_padding_idx)
    initialize_model(model.src2trg_translator, cfg, src_padding_idx, trg_padding_idx)
    initialize_model(model.trg2src_translator, cfg, trg_padding_idx, src_padding_idx)
    initialize_model(model.trg2src_translator, cfg, trg_padding_idx, trg_padding_idx)
    return model
