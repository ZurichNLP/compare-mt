
import itertools
import numpy as np

from typing import List, Tuple, Optional
from collections import Counter

from compare_mt import ngram_utils


Tokens = List[str]
Sentences = List[Tokens]


def repetition_stats(ref: Sentences,
                     outs: List[Sentences],
                     src: Optional[Sentences] = None,
                     adjacent: bool = True,
                     ngram_order: int = 1,
                     subtract_legitimate_reps: bool = False) -> List[int]:
    """
    For each sentence translated by a system, collects the number of repeated elements.

    :param ref: Lines from a reference corpus, in the target language.
    :param outs: Lines from several system hypotheses.
    :param src: Lines from a reference corpus, in the source language.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.
    :param subtract_legitimate_reps: Whether to account for repetitions that are actually
                                     present in the ref or src data.

    :return: Lists of integers, where list elements correspond to lines in an input corpus.
    """

    rep_totals_per_out = []  # type: List[int]

    for out in outs:
        reps_per_line = repetition_stats_in_corpus(ref, out, src, adjacent,
                                                   ngram_order, subtract_legitimate_reps)

        rep_total = sum(reps_per_line)
        rep_totals_per_out.append(rep_total)

    return rep_totals_per_out


def repetition_stats_in_corpus(ref: Sentences,
                               out: Sentences,
                               src: Optional[Sentences] = None,
                               adjacent: bool = True,
                               ngram_order: int = 1,
                               subtract_legitimate_reps: bool = False) -> List[int]:
    """
    For each sentence translated by a system, collects the number of repeated elements.

    :param ref: Lines from a reference corpus, in the target language.
    :param out: Translations from a single system.
    :param src: Lines from a reference corpus, in the source language.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.
    :param subtract_legitimate_reps: Whether to account for repetitions that are actually
                                     present in the ref or src data.

    :return: A list of integers, where list elements correspond to lines in the input corpus.
    """

    reps_per_line = []

    src_lines = [] if src is None else src

    for out_line, ref_line, src_line in itertools.zip_longest(out, ref, src_lines):

        out_reps = num_repetitions_in_sentence(out_line, adjacent, ngram_order)

        if subtract_legitimate_reps:
            ref_reps = num_repetitions_in_sentence(ref_line, adjacent, ngram_order)

            if src is None:
                src_reps = 0
            else:
                src_reps = num_repetitions_in_sentence(src_line, adjacent, ngram_order)

            out_reps -= max(src_reps, ref_reps)

        reps_per_line.append(out_reps)

    return reps_per_line


def repetition_examples(ref: Sentences,
                        outs: List[Sentences],
                        src: Optional[Sentences] = None,
                        num_examples: int = 10,
                        adjacent: bool = True,
                        ngram_order: int = 1,
                        ignore_legitimate_reps: bool = False
                        ) -> List[List[int]]:
    """
    Find the indexes of the worst examples of repeated material in
    each set of system translations (outs).

    :param ref: Lines from a reference corpus, in the target language.
    :param outs: Lines from several system hypotheses.
    :param src: Lines from a reference corpus, in the source language.
    :param num_examples: Number of example sentences to find.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.
    :param ignore_legitimate_reps: Whether to account for repetitions that are actually
                                   present in the ref or src data.

    :return:
    """

    indexes_per_out = []

    for out in outs:
        indexes = repetition_examples_from_corpus(out, ref, src, num_examples, adjacent,
                                                   ngram_order, ignore_legitimate_reps)
        indexes_per_out.append(indexes)

    return indexes_per_out


def repetition_examples_from_corpus(out: Sentences,
                                    ref: Sentences,
                                    src: Optional[Sentences] = None,
                                    num_examples: int = 10,
                                    adjacent: bool = True,
                                    ngram_order: int = 1,
                                    ignore_legitimate_reps: bool = False
                                    ) -> List[int]:
    """
    Find the indexes of the worst examples of repeated material in a corpus (out).

    :param out: Translations from a single system.
    :param ref: Lines from a reference corpus, in the target language.
    :param src: Lines from a reference corpus, in the source language.
    :param num_examples: Number of example sentences to find.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.
    :param ignore_legitimate_reps: Whether to account for repetitions that are actually
                                   present in the ref or src data.

    :return:
    """

    rep_stats = repetition_stats_in_corpus(ref, out, src, adjacent,
                                           ngram_order, ignore_legitimate_reps)

    # reverse sort index
    sort_index = np.argsort(rep_stats, )[::-1]

    # truncate to length of report
    trunc_index = sort_index[:num_examples]

    return trunc_index.tolist()


def num_repetitions_in_sentence(sentence: Tokens,
                                adjacent: bool = True,
                                ngram_order: int = 1) -> int:
    """
    Counts repetitions in an input sentence.

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
                                     ngram_order: int = 1) -> Tuple[int, int]:
    """

    :param src_sentence: Line in source language.
    :param trg_sentence: Line in target language.
    :param adjacent: Whether repeated elements need to occur adjacent
                     to each other to count towards repetitions.
    :param ngram_order: Order of ngrams considered, positive integer.

    :return: Number of repetitions in each sentence, as a tuple of integers.
    """

    src_reps = num_repetitions_in_sentence(src_sentence, adjacent, ngram_order)
    trg_reps = num_repetitions_in_sentence(trg_sentence, adjacent, ngram_order)

    return src_reps, trg_reps
