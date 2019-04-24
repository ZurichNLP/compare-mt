import os.path
import unittest
import sys

compare_mt_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(compare_mt_root)

from compare_mt.repetition_utils import num_repetitions_in_sentence


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