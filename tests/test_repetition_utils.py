import os.path
import unittest
import sys

compare_mt_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(compare_mt_root)

from compare_mt.repetition_utils import num_repetitions_in_sentence, repetition_stats_in_corpus, repetition_examples_from_corpus


class TestRepetitionUtils(unittest.TestCase):

    def test_num_repetitions_in_sentence_unigram_no_reps(self):

        sentence = "This is a test !".split(" ")
        expected_reps = 0

        actual_reps = num_repetitions_in_sentence(sentence=sentence,
                                                  adjacent=True,
                                                  ngram_order=1)

        self.assertEqual(expected_reps, actual_reps, "Wrong number of repetitions detected.")

    def test_num_repetitions_in_sentence_unigram_no_adjacent_reps(self):

        sentence = "a This is a test ! a".split(" ")
        expected_reps = 0

        actual_reps = num_repetitions_in_sentence(sentence=sentence,
                                                  adjacent=True,
                                                  ngram_order=1)

        self.assertEqual(expected_reps, actual_reps, "Wrong number of repetitions detected.")

        expected_reps = 2

        actual_reps = num_repetitions_in_sentence(sentence=sentence,
                                                  adjacent=False,
                                                  ngram_order=1)

        self.assertEqual(expected_reps, actual_reps, "Wrong number of repetitions detected.")

    def test_num_repetitions_in_sentence_unigram_adjacent_reps(self):

        sentence = "This is a a a test !".split(" ")
        expected_reps = 2

        actual_reps = num_repetitions_in_sentence(sentence=sentence,
                                                  adjacent=True,
                                                  ngram_order=1)

        self.assertEqual(expected_reps, actual_reps, "Wrong number of repetitions detected.")

    def test_num_repetitions_in_sentence_bigram_adjacent_reps(self):

        sentence = "This is a is a test !".split(" ")
        expected_reps = 1

        actual_reps = num_repetitions_in_sentence(sentence=sentence,
                                                  adjacent=True,
                                                  ngram_order=2)

        self.assertEqual(expected_reps, actual_reps, "Wrong number of repetitions detected.")

    def test_repetition_stats_in_corpus_unigram(self):

        ref = ["This is the first test sentence".split(" "),
               "This is the second test sentence".split(" ")]

        out = ["This is is the first test sentence".split(" "),
               "This is the the the the the second test is sentence".split(" ")]

        expected = [1, 4]

        actual = repetition_stats_in_corpus(ref=ref,
                                            out=out,
                                            src=None,
                                            adjacent=True,
                                            ngram_order=1,
                                            subtract_legitimate_reps=False)

        self.assertEqual(expected, actual, "Wrong number of repetitions detected.")

        # also count non-adjacent repetitions

        expected = [1, 5]

        actual = repetition_stats_in_corpus(ref=ref,
                                            out=out,
                                            src=None,
                                            adjacent=False,
                                            ngram_order=1,
                                            subtract_legitimate_reps=False)

        self.assertEqual(expected, actual, "Wrong number of repetitions detected.")

        # subtract legitimate reps in ref

        ref_with_reps = ["This is is the first test sentence".split(" "),
                         "This is the second test sentence".split(" ")]

        expected = [0, 4]

        actual = repetition_stats_in_corpus(ref=ref_with_reps,
                                            out=out,
                                            src=None,
                                            adjacent=True,
                                            ngram_order=1,
                                            subtract_legitimate_reps=True)

        self.assertEqual(expected, actual, "Wrong number of repetitions detected.")

    def test_repetition_examples_from_corpus(self):
        ref = ["This is the first test sentence".split(" "),
               "This is the second test sentence".split(" "),
               "This is the third test sentence".split(" ")]

        out = ["This is is is is is is is the first test sentence".split(" "),
               "This is the second test sentence".split(" "),
               "This is the third third third test sentence".split(" ")]

        expected = [0, 2]

        actual = repetition_examples_from_corpus(out=out,
                                                 ref=ref,
                                                 src=None,
                                                 num_examples=2,
                                                 adjacent=True,
                                                 ngram_order=1,
                                                 ignore_legitimate_reps=False)

        self.assertEqual(expected, actual, "Wrong worst indexes from corpus.")
