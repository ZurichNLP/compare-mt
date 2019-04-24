
import itertools

from typing import List, Tuple
from collections import Counter

from compare_mt import ngram_utils


Tokens = List[str]
Sentences = List[Tokens]


def repetition_stats(ref: Sentences,
                     outs: List[Sentences],
                     src: Sentences = None,
                     adjacent: bool = True,
                     ngram_order: int = 1,
                     subtract_legitimate_reps: bool = False):
    """

    :param ref:
    :param outs:
    :param src:
    :param adjacent:
    :param ngram_order:
    :param subtract_legitimate_reps:
    :return:
    """

    reps_totals = []

    if src is None:
        src_lines = []

    for out_lines in outs:
        reps = 0

        for out_line, ref_line, src_line in itertools.zip_longest(out_lines, ref, src_lines):

            out_reps = num_repetitions_in_sentence(out_line, adjacent, ngram_order)

            if subtract_legitimate_reps:
                ref_reps = num_repetitions_in_sentence(ref_line, adjacent, ngram_order)

                if src is None:
                    src_reps = 0
                else:
                    src_reps = num_repetitions_in_sentence(src_line, adjacent, ngram_order)

                out_reps -= max(src_reps, ref_reps)

            reps += out_reps

        reps_totals.append(reps)

    return reps_totals



def repetition_examples(ref: Sentences,
                        outs: List[Sentences],
                        src: Sentences = None,
                        report_length: int = 10):
    """

    :param ref:
    :param outs:
    :param src:
    :param report_length:
    :return:
    """
    pass

def num_repetitions_in_sentence(sentence: Tokens,
                                adjacent: bool = True,
                                ngram_order: int = 1) -> int:
    """

    :param sentence: A list of tokens or characters as strings.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.
    :return: Number of times an element was repeated.
    """
    num_repetitions = 0

    ngrams = ngram_utils.sent_ngrams_list(sentence, ngram_order)

    if not adjacent:
        counter = Counter(ngrams)
        for k, v in counter.items():
            num_repetitions += (v - 1)

    else:
        previous = []

        for ngram in ngrams:
            if ngram in previous:
                num_repetitions += 1
            previous.append(ngram)

            if len(previous) > ngram_order:
                previous.pop(0)

    return num_repetitions

def num_repetitions_in_sentence_pair(src_sentence: Tokens,
                                     trg_sentence: Tokens,
                                     adjacent: bool = True,
                                     ngram_order: int = 1) -> Tuple[int,int]:
    """

    :param src_sentence:
    :param trg_sentence:
    :param adjacent:
    :param ngram_order:
    :return:
    """

    src_reps = num_repetitions_in_sentence(src_sentence, adjacent, ngram_order)
    trg_reps = num_repetitions_in_sentence(trg_sentence, adjacent, ngram_order)

    return src_reps, trg_reps