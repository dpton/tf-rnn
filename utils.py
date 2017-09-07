# _*_ coding:utf-8 _*_
# !/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import logging
import nltk
from text2num import text2num
from nltk.corpus import stopwords
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
import random
import threading
import tensorflow as tf



import re
import unicodedata

from ftfy import fix_text
from unidecode import unidecode

from textacy.compat import unicode_
from textacy.constants import (CURRENCIES, URL_REGEX, SHORT_URL_REGEX, EMAIL_REGEX,
                               PHONE_REGEX, NUMBERS_REGEX, CURRENCY_REGEX,
                               LINEBREAK_REGEX, NONBREAKING_SPACE_REGEX,
                               PUNCT_TRANSLATE_UNICODE,
                               PUNCT_TRANSLATE_BYTES)


UNK = u'_UNK'
GO = u'_GO'
EOS = u'_EOS'
PAD = u'_PAD'
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_WORDS = [PAD, GO, EOS, UNK]


def fix_bad_unicode(text, normalization='NFC'):
    """
    Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
    this includes mojibake, HTML entities and other code cruft,
    and non-standard forms for display purposes.

    Args:
        text (str): raw text
        normalization ({'NFC', 'NFKC', 'NFD', 'NFKD'}): if 'NFC',
            combines characters and diacritics written using separate code points,
            e.g. converting "e" plus an acute accent modifier into "é"; unicode
            can be converted to NFC form without any change in its meaning!
            if 'NFKC', additional normalizations are applied that can change
            the meanings of characters, e.g. ellipsis characters will be replaced
            with three periods

    Returns:
        str
    """
    return fix_text(text, normalization=normalization)


def transliterate_unicode(text):
    """
    Try to represent unicode data in ascii characters similar to what a human
    with a US keyboard would choose using unidecode <https://pypi.python.org/pypi/Unidecode>
    """
    return unidecode(text)


def normalize_whitespace(text):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more linebreaks with a single newline. Also strip leading/trailing whitespace.
    """
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()


def unpack_contractions(text):
    """
    Replace *English* contractions in ``text`` str with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.
    """
    # standard
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(
        r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text


def replace_urls(text, replace_with='<url>'):
    """Replace all URLs in ``text`` str with ``replace_with`` str."""
    return URL_REGEX.sub(replace_with, SHORT_URL_REGEX.sub(replace_with, text))


def replace_emails(text, replace_with='<email>'):
    """Replace all emails in ``text`` str with ``replace_with`` str."""
    return EMAIL_REGEX.sub(replace_with, text)


def replace_phone_numbers(text, replace_with='<phone>'):
    """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
    return PHONE_REGEX.sub(replace_with, text)


def replace_numbers(text, replace_with='<number>'):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return NUMBERS_REGEX.sub(replace_with, text)


def replace_currency_symbols(text, replace_with='<currency>'):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.

    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")

    Returns:
        str
    """
    if replace_with is None:
        for k, v in CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return CURRENCY_REGEX.sub(replace_with, text)


def remove_punct(text, marks=None):
    """
    Remove punctuation from ``text`` by replacing all instances of ``marks``
    with an empty string.

    Args:
        text (str): raw text
        marks (str): If specified, remove only the characters in this string,
            e.g. ``marks=',;:'`` removes commas, semi-colons, and colons.
            Otherwise, all punctuation marks are removed.

    Returns:
        str

    .. note:: When ``marks=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise,, a regular expression is used
        instead. The former's performance is about 5-10x faster.
    """
    if marks:
        return re.sub('[{}]+'.format(re.escape(marks)), '', text, flags=re.UNICODE)
    else:
        if isinstance(text, unicode_):
            return text.translate(PUNCT_TRANSLATE_UNICODE)
        else:
            return text.translate(None, PUNCT_TRANSLATE_BYTES)


def remove_accents(text, method='unicode'):
    """
    Remove accents from any accented unicode characters in ``text`` str, either by
    transforming them into ascii equivalents or removing them entirely.

    Args:
        text (str): raw text
        method ({'unicode', 'ascii'}): if 'unicode', remove accented
            char for any unicode symbol with a direct ASCII equivalent; if 'ascii',
            remove accented char for any unicode symbol

            NB: the 'ascii' method is notably faster than 'unicode', but less good

    Returns:
        str

    Raises:
        ValueError: if ``method`` is not in {'unicode', 'ascii'}
    """
    if method == 'unicode':
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))
    elif method == 'ascii':
        return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii')
    else:
        msg = '`method` must be either "unicode" and "ascii", not {}'.format(
            method)
        raise ValueError(msg)


def replace_hashtag(text, replace_with_hashtag='<hashtag>'):
    text = ' '.join(
        re.sub("(#[A-Za-z0-9]+)", replace_with_hashtag, text).split())
    return text


def replace_nametag(text, replace_with_nametag='<nametag>'):
    text = ' '.join(
        re.sub("(@[A-Za-z0-9]+)", replace_with_nametag, text).split())
    return text


def preprocess_text(text, fix_unicode=False, lowercase=False, transliterate=False,
                    no_urls=False, no_emails=False, no_phone_numbers=False,
                    no_numbers=False, no_currency_symbols=False, no_punct=False,
                    no_contractions=False, no_accents=False, no_hashtag=False, no_nametag=False):
    """
    This is a modified version of the function "textacy.preprocess_text".
    Normalize various aspects of a raw text doc before parsing it with Spacy.
    A convenience function for applying all other preprocessing functions in one go.

    Args:
        text (str): raw text to preprocess
        fix_unicode (bool): if True, fix "broken" unicode such as
            mojibake and garbled HTML entities
        lowercase (bool): if True, all text is lower-cased
        transliterate (bool): if True, convert non-ascii characters
            into their closest ascii equivalents
        no_urls (bool): if True, replace all URL strings with '*URL*'
        no_emails (bool): if True, replace all email strings with '*EMAIL*'
        no_phone_numbers (bool): if True, replace all phone number strings
            with '<phone>'
        no_numbers (bool): if True, replace all number-like strings
            with '<number>'
        no_currency_symbols (bool): if True, replace all currency symbols
            with their standard 3-letter abbreviations
        no_punct (bool): if True, remove all punctuation (replace with
            empty string)
        no_contractions (bool): if True, replace *English* contractions
            with their unshortened forms
        no_accents (bool): if True, replace all accented characters
            with unaccented versions; NB: if `transliterate` is True, this option
            is redundant
        not_hashtag (bool): if True, replace all hashtag (twitter, facebook)

    Returns:
        str: input ``text`` processed according to function args

    .. warning:: These changes may negatively affect subsequent NLP analysis
        performed on the text, so choose carefully, and preprocess at your own
        risk!
    """
    if fix_unicode is True:
        text = fix_bad_unicode(text, normalization='NFC')
    if transliterate is True:
        text = transliterate_unicode(text)
    if no_urls is True:
        text = replace_urls(text)
    if no_emails is True:
        text = replace_emails(text)
    if no_phone_numbers is True:
        text = replace_phone_numbers(text)
    if no_numbers is True:
        text = replace_numbers(text)
    if no_currency_symbols is True:
        text = replace_currency_symbols(text)
    if no_contractions is True:
        text = unpack_contractions(text)
    if no_accents is True:
        text = remove_accents(text, method='unicode')
    if no_punct is True:
        text = remove_punct(text)
    if lowercase is True:
        text = text.lower()
    if no_hashtag is True:
        text = replace_hashtag(text)
    if no_nametag is True:
        text = replace_nametag(text)
    # always normalize whitespace; treat linebreaks separately from spacing
    text = normalize_whitespace(text)

    return text


def text_normalize(string, convert2digit=True):
    text = preprocess_text(text=string, fix_unicode=False, lowercase=True, transliterate=False,
                           no_urls=True, no_emails=True, no_phone_numbers=True,
                           no_numbers=True, no_currency_symbols=True, no_punct=False,
                           no_contractions=True, no_accents=True, not_hashtag=True)
    if convert2digit:
        return text2num(text)
    else:
        return text


def remove_stopwords(word_list):
    filtered_words = [
        word for word in word_list if word not in stopwords.words('english')]
    return filtered_words


def tokenize(string):
    text = text_normalize(string)
    return nltk.word_tokenize(text, language='english')


class wordEmbedding(object):
    '''
    This class wraps the two popular models using for word embedding, FastText and Word2Vec
    '''

    def __init__(self, model_path, model_type='fasttext', **kwarg):
        if model_type == "fasttext":
            self._model = FastText.load_fasttext_format(model_path)
        elif model_type == "word2vec":
            self._model = Word2Vec.load_word2vec_format(model_path)
        else:
            raise NotImplementedError("other model is not supported")

    def sentence_to_index(self, sentence):
        list_of_index = [self._model.wv.vocab[
            word].index for word in tokenize(sentence)]
        return list_of_index

    def get_embedding_matrix(self):
        return self._model.syn0

def create_queue(sess = None, coord = None, encode_data = None,
                 decode_data = None, capacity = 1024, batch_size = 32, scope = None):

    encode = tf.placeholder(tf.int32, shape=[None], name="encode")
    decode = tf.placeholder(tf.int32, shape=[decode_max_length + 2], name="decode")
    weight = tf.placeholder(tf.float32, shape=[decode_max_length + 1], name="weight")
    queue = tf.PaddingFIFOQueue(capacity = capacity,
                        dtypes = [tf.int32, tf.int32, tf.float32],
                        shapes = [[None], [decode_max_length + 2], [decode_max_length + 1]],
                        name = 'FIFOQueue')
    enqueue_op = queue.enqueue([encode, decode, weight])


    def _iterator():
        assert len(encode_data) == len(decode_data)
        data = list(zip(encode_data, decode_data))
        random.shuffle(data)
        encode, decode = [list(t) for t in zip(*data)]

        for i in range(len(data)):
#             if len(encode[i]) > encode_max_length - 1 or len(decode[i]) > decode_max_length - 1:
#                 raise 'the sentence is longer than max_length'
            #_encode = encode[i][:encode_max_length]
            #_encode = _encode + [PAD_ID] * (encode_max_length - len(_encode))
            _encode = encode[i]
            _decode = decode[i][:decode_max_length]
            
        
            
            _decode_padding_size = decode_max_length - len(_decode)
            _weight = [1.0] * (len(_decode) + 1) + [0.0] * _decode_padding_size
            _decode = [GO_ID] + _decode + [EOS_ID] + [PAD_ID] * _decode_padding_size
            
            yield _encode, _decode, _weight#, _encode_length, _decode_length
    def basic_enqueue(sess, encode_input, decode_input = None):
#         if len(encode_input) > encode_max_length:
#             encode_input = encode_input[:encode_max_length]
#         _encode = encode_input + [PAD_ID] * (encode_max_length - len(encode_input))
        _encode = encode_input
        if decode_input is None:
            _decode = [GO_ID] + [PAD_ID] * (decode_max_length + 1)
            _weight = [1.0] * (decode_max_length + 1)
        else:
            _decode_padding_size = decode_max_length - len(decode_input)
            _decode = [GO_ID] + decode_input + [EOS_ID] + [PAD_ID] * _decode_padding_size
            _weight = [1.0] * (len(decode_input) + 1) + [0.0] * _decode_padding_size
        feed_dict = {
                encode: _encode,
                decode: _decode,
                weight: _weight
            }
        # Push all the training examples to the queue
        sess.run(enqueue_op, feed_dict=feed_dict)
    def _enqueue(sess, coord):
        try:
            while not coord.should_stop():
                for _encode, _decode, _weight in _iterator():
                    feed_dict = {
                        encode: _encode,
                        decode: _decode,
                        weight: _weight,
                    }
                    # Push all the training examples to the queue
                    sess.run(enqueue_op, feed_dict=feed_dict)
        except tf.errors.CancelledError:
            coord.request_stop()
    #Start thread enqueue data
    # if encode_data is None or  decode_data is None:
    #     return queue, None, basic_enqueue
    enqueue_threads = []
    ## enqueue asynchronously
    for i in range(num_threads):
        enqueue_thread = threading.Thread(target=_enqueue, args=(sess, coord))
        enqueue_thread.setDaemon(True)
        enqueue_threads.append(enqueue_thread)
    return queue, enqueue_threads, basic_enqueue

if __name__ == '__main__':
    print(tokenize("http://google.com.vn I love the cat @Peter with 69USD"))
