# Overall imports
import argparse
import operator
from whatthelang import WhatTheLang
import langid
from collections import defaultdict

# In-package imports
from compare_mt import ngram_utils
from compare_mt import stat_utils
from compare_mt import corpus_utils
from compare_mt import sign_utils
from compare_mt import scorers
from compare_mt import bucketers
from compare_mt import reporters
from compare_mt import arg_utils
from compare_mt import formatting
from compare_mt import repetition_utils

def generate_score_report(ref, outs,
                       score_type='bleu',
                       bootstrap=0, prob_thresh=0.05,
                       meteor_directory=None, options=None,
                       title=None, 
                       case_insensitive=False):
  """
  Generate a report comparing overall scores of system(s) in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    score_type: A string specifying the scoring type (bleu/length)
    bootstrap: Number of samples for significance test (0 to disable)
    prob_thresh: P-value threshold for significance test
    meteor_directory: Path to the directory of the METEOR code
    options: Options when using external program
    compare_directions: A string specifying which systems to compare 
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  bootstrap = int(bootstrap)
  prob_thresh = float(prob_thresh)
  case_insensitive = True if case_insensitive == 'True' else False

  scorer = scorers.create_scorer_from_profile(score_type, case_insensitive=case_insensitive, meteor_directory=meteor_directory, options=options)

  scores, strs = zip(*[scorer.score_corpus(ref, out) for out in outs])

  if bootstrap != 0:
    direcs = []
    for i in range(len(scores)):
      for j in range(i+1, len(scores)):
        direcs.append( (i,j) )
    wins, sys_stats = sign_utils.eval_with_paired_bootstrap(ref, outs, scorer, direcs, num_samples=bootstrap)
    wins = list(zip(direcs, wins))
  else:
    wins = sys_stats = direcs = None

  reporter = reporters.ScoreReport(scorer=scorer, scores=scores, strs=strs, 
                                   wins=wins, sys_stats=sys_stats, prob_thresh=prob_thresh, 
                                   title=title)
  reporter.generate_report(output_fig_file=f'score-{score_type}-{bootstrap}',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 

def generate_word_accuracy_report(ref, outs,
                          acc_type='fmeas', bucket_type='freq', bucket_cutoffs=None,
                          freq_count_file=None, freq_corpus_file=None,
                          label_set=None,
                          ref_labels=None, out_labels=None,
                          title=None,
                          case_insensitive=False):
  """
  Generate a report comparing the word accuracy in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    acc_type: The type of accuracy to show (prec/rec/fmeas). Can also have multiple separated by '+'.
    bucket_type: A string specifying the way to bucket words together to calculate F-measure (freq/tag)
    bucket_cutoffs: The boundaries between buckets, specified as a colon-separated string.
    freq_corpus_file: When using "freq" as a bucketer, which corpus to use to calculate frequency.
                      By default this uses the frequency in the reference test set, but it's often more informative
                      to use the frequency in the training set, in which case you specify the path of the
                      training corpus.
    freq_count_file: An alternative to freq_corpus that uses a count file in "word\tfreq" format.
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
    out_labels: output labels. must be specified if ref_labels is specified.
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  case_insensitive = True if case_insensitive == 'True' else False

  if out_labels is not None:
    out_labels = arg_utils.parse_files(out_labels)
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')

  bucketer = bucketers.create_word_bucketer_from_profile(bucket_type,
                                                         bucket_cutoffs=bucket_cutoffs,
                                                         freq_count_file=freq_count_file,
                                                         freq_corpus_file=freq_corpus_file,
                                                         freq_data=ref,
                                                         label_set=label_set,
                                                         case_insensitive=case_insensitive)
  ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
  out_labels = [corpus_utils.load_tokens(out_labels[i]) if not out_labels is None else None for i in range(len(outs))]
  matches = [bucketer.calc_bucketed_matches(ref, out, ref_labels=ref_labels, out_labels=out_label) for out, out_label in zip(outs, out_labels)]
  
  reporter = reporters.WordReport(bucketer=bucketer, matches=matches,
                                  acc_type=acc_type, header="Word Accuracy Analysis", 
                                  title=title)
  reporter.generate_report(output_fig_file=f'word-acc',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 
  

def generate_src_word_accuracy_report(ref, outs, src, ref_align_file=None, out_align_files=None,
                          acc_type='fmeas', bucket_type='freq', bucket_cutoffs=None,
                          freq_count_file=None, freq_corpus_file=None,
                          label_set=None,
                          src_labels=None,
                          title=None,
                          case_insensitive=False):
  """
  Generate a report for source word analysis in both plain text and graphs.

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens from the source
    ref_align_file: Alignment file for the reference
    out_align_files: Alignment file for the output file
    acc_type: The type of accuracy to show (prec/rec/fmeas). Can also have multiple separated by '+'.
    bucket_type: A string specifying the way to bucket words together to calculate F-measure (freq/tag)
    bucket_cutoffs: The boundaries between buckets, specified as a colon-separated string.
    freq_corpus_file: When using "freq" as a bucketer, which corpus to use to calculate frequency.
                      By default this uses the frequency in the reference test set, but it's often more informative
                      se the frequency in the training set, in which case you specify the path of the target side
                      he training corpus.
    freq_count_file: An alternative to freq_corpus that uses a count file in "word\tfreq" format.
    src_labels: either a filename of a file full of source labels, or a list of strings corresponding to `ref`.
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  case_insensitive = True if case_insensitive == 'True' else False

  if not src or not ref_align_file or not out_align_files:
    raise ValueError("Must specify the source and the alignment files when performing source analysis.")

  ref_align = corpus_utils.load_tokens(ref_align_file) 
  out_aligns = [corpus_utils.load_tokens(x) for x in arg_utils.parse_files(out_align_files)]

  if len(out_aligns) != len(outs):
    raise ValueError(f'The number of output files should be equal to the number of output alignment files.')

  bucketer = bucketers.create_word_bucketer_from_profile(bucket_type,
                                                         bucket_cutoffs=bucket_cutoffs,
                                                         freq_count_file=freq_count_file,
                                                         freq_corpus_file=freq_corpus_file,
                                                         freq_data=src,
                                                         label_set=label_set,
                                                         case_insensitive=case_insensitive)
  src_labels = corpus_utils.load_tokens(src_labels) if type(src_labels) == str else src_labels
  matches = [bucketer.calc_source_bucketed_matches(src, ref, out, ref_align, out_align, src_labels=src_labels) for out, out_align in zip(outs, out_aligns)]

  reporter = reporters.WordReport(bucketer=bucketer, matches=matches,
                                  acc_type=acc_type, header="Source Word Accuracy Analysis", 
                                  title=title)
  reporter.generate_report(output_fig_file=f'src-word-acc',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 

def generate_sentence_bucketed_report(ref, outs,
                                   bucket_type='score', bucket_cutoffs=None,
                                   statistic_type='count',
                                   score_measure='bleu',
                                   label_set=None,
                                   ref_labels=None, out_labels=None,
                                   title=None,
                                   case_insensitive=False):
  """
  Generate a report of sentences by bucket in both plain text and graphs

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    bucket_type: The type of bucketing method to use
    score_measure: If using 'score' as either bucket_type or statistic_type, which scorer to use
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`. Would overwrite out_labels if specified.
    out_labels: output labels. 
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  case_insensitive = True if case_insensitive == 'True' else False

  if ref_labels is not None:
    ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
    if len(ref_labels) != len(ref):
      raise ValueError(f'The number of labels should be equal to the number of sentences.')

  elif out_labels is not None:
    out_labels = arg_utils.parse_files(out_labels)
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')

    out_labels = [corpus_utils.load_tokens(out_label) if type(out_label) == str else out_label for out_label in out_labels]
    for out, out_label in zip(outs, out_labels):
      if len(out_label) != len(out):
        raise ValueError(f'The number of labels should be equal to the number of sentences.')
    

  bucketer = bucketers.create_sentence_bucketer_from_profile(bucket_type, bucket_cutoffs=bucket_cutoffs,
                                                             score_type=score_measure, label_set=label_set, case_insensitive=case_insensitive)
  bcs = [bucketer.create_bucketed_corpus(out, ref=ref, ref_labels=ref_labels if ref_labels else None, out_labels=out_labels[i] if out_labels else None) for i, out in enumerate(outs)]

  if statistic_type == 'count':
    scorer = None
    aggregator = lambda out,ref: len(out)
  elif statistic_type == 'score':
    scorer = scorers.create_scorer_from_profile(score_measure, case_insensitive=case_insensitive)
    aggregator = lambda out,ref: scorer.score_corpus(ref,out)[0]
  else:
    raise ValueError(f'Illegal statistic_type {statistic_type}')

  stats = [[aggregator(out,ref) for (out,ref) in bc] for bc in bcs]

  reporter = reporters.SentenceReport(bucketer=bucketer,
                                      sys_stats=stats,
                                      statistic_type=statistic_type, scorer=scorer, 
                                      title=title)

  reporter.generate_report(output_fig_file=f'sentence-{statistic_type}-{score_measure}',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 
  

def generate_ngram_report(ref, outs,
                       min_ngram_length=1, max_ngram_length=4,
                       report_length=50, alpha=1.0, compare_type='match',
                       ref_labels=None, out_labels=None,
                       compare_directions='0-1',
                       title=None,
                       case_insensitive=False):
  """
  Generate a report comparing aggregate n-gram statistics in both plain text and graphs

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    min_ngram_length: minimum n-gram length
    max_ngram_length: maximum n-gram length
    report_length: the number of n-grams to report
    alpha: when sorting n-grams for salient features, the smoothing coefficient. A higher smoothing coefficient
           will result in more frequent phenomena (sometimes this is good).
    compare_type: what type of statistic to compare
                  (match: n-grams that match the reference, over: over-produced ngrams, under: under-produced ngrams)
    ref_labels: either a filename of a file full of reference labels, or a list of strings corresponding to `ref`.
                If specified, will aggregate statistics over labels instead of n-grams.
    out_labels: output labels. must be specified if ref_labels is specified.
    compare_directions: A string specifying which systems to compare
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  min_ngram_length, max_ngram_length, report_length = int(min_ngram_length), int(max_ngram_length), int(report_length)
  alpha = float(alpha)
  case_insensitive = True if case_insensitive == 'True' else False

  if out_labels is not None:
    out_labels = arg_utils.parse_files(out_labels)
    if len(out_labels) != len(outs):
      raise ValueError(f'The number of output files should be equal to the number of output labels.')

  if type(ref_labels) == str:
    label_files_str = f'    ref_labels={ref_labels},'
    for i, out_label in enumerate(out_labels):
      label_files_str += f' out{i}_labels={out_label},'
    label_files = (label_files_str)
  else:
    label_files = None

  if type(alpha) == str:
    alpha = float(alpha)

  if not type(ref_labels) == str and case_insensitive:
    ref = corpus_utils.lower(ref)
    outs = [corpus_utils.lower(out) for out in outs]

  ref_labels = corpus_utils.load_tokens(ref_labels) if type(ref_labels) == str else ref_labels
  out_labels = [corpus_utils.load_tokens(out_labels[i]) if not out_labels is None else None for i in range(len(outs))]
  totals, matches, overs, unders = zip(*[ngram_utils.compare_ngrams(ref, out, ref_labels=ref_labels, out_labels=out_label,
                                                             min_length=min_ngram_length, max_length=max_ngram_length) for out, out_label in zip(outs, out_labels)])
  direcs = arg_utils.parse_compare_directions(compare_directions)
  scores = []
  for (left, right) in direcs:
    if compare_type == 'match':
      scores.append(stat_utils.extract_salient_features(matches[left], matches[right], alpha=alpha))
    elif compare_type == 'over':
      scores.append(stat_utils.extract_salient_features(overs[left], overs[right], alpha=alpha))
    elif compare_type == 'under':
      scores.append(stat_utils.extract_salient_features(unders[left], unders[right], alpha=alpha))
    else:
      raise ValueError(f'Illegal compare_type "{compare_type}"')
  scorelist = [sorted(score.items(), key=operator.itemgetter(1), reverse=True) for score in scores]

  reporter = reporters.NgramReport(scorelist=scorelist, report_length=report_length,
                                   min_ngram_length=min_ngram_length, 
                                   max_ngram_length=max_ngram_length,
                                   matches=matches,
                                   compare_type=compare_type, alpha=alpha,
                                   compare_directions=direcs,
                                   label_files=label_files,
                                   title=title)                                   
  reporter.generate_report(output_fig_file=f'ngram-min{min_ngram_length}-max{max_ngram_length}-{compare_type}',
                           output_fig_format='pdf', 
                           output_directory='outputs')
  return reporter 

def generate_sentence_examples(ref, outs, src=None,
                            score_type='sentbleu',
                            report_length=10,
                            compare_directions='0-1',
                            title=None,
                            case_insensitive=False):
  """
  Generate examples of sentences that satisfy some criterion, usually score of one system better

  Args:
    ref: Tokens from the reference
    outs: Tokens from the output file(s)
    src: Tokens from the source (optional)
    score_type: The type of scorer to use
    report_length: Number of sentences to print for each system being better or worse
    compare_directions: A string specifying which systems to compare
    title: A string specifying the caption of the printed table
    case_insensitive: A boolean specifying whether to turn on the case insensitive option
  """
  report_length = int(report_length)
  case_insensitive = True if case_insensitive == 'True' else False
    
  scorer = scorers.create_scorer_from_profile(score_type, case_insensitive=case_insensitive)

  direcs = arg_utils.parse_compare_directions(compare_directions)

  scorediff_lists = []
  for (left, right) in direcs:
    scorediff_list = []
    deduplicate_set = set()
    for i, (o1, o2, r) in enumerate(zip(outs[left], outs[right], ref)):
      if (tuple(o1), tuple(o2), tuple(r)) in deduplicate_set:
        continue
      deduplicate_set.add( (tuple(o1), tuple(o2), tuple(r)) )
      s1, str1 = scorer.score_sentence(r, o1)
      s2, str2 = scorer.score_sentence(r, o2)
      scorediff_list.append((s2-s1, s1, s2, str1, str2, i))
    scorediff_list.sort()
    scorediff_lists.append(scorediff_list)

  reporter = reporters.SentenceExampleReport(report_length=report_length, scorediff_lists=scorediff_lists,
                                             scorer=scorer,
                                             ref=ref, outs=outs, src=src,
                                             compare_directions=direcs,
                                             title=title)
  reporter.generate_report()
  return reporter


def generate_repetitions_report(ref, outs, src=None, title=None, adjacent=True,
                                ngram_order=1, subtract_legitimate_reps=False):

    ngram_order = int(ngram_order)

    repetition_stats = repetition_utils.repetition_stats(ref=ref,
                                                         outs=outs,
                                                         src=src,
                                                         adjacent=adjacent,
                                                         ngram_order=ngram_order,
                                                         subtract_legitimate_reps=subtract_legitimate_reps)

    reporter = reporters.RepetitionReport(ref=ref,
                                          outs=outs,
                                          src=src,
                                          repetition_stats=repetition_stats,
                                          title=title,
                                          adjacent=adjacent,
                                          ngram_order=ngram_order,
                                          subtract_legitimate_reps=subtract_legitimate_reps)

    reporter.generate_report()

    return reporter


def generate_repetitions_examples(ref, outs, src=None, report_length=10, title=None, adjacent=True,
                                  ngram_order=1, ignore_legitimate_reps=False):

    report_length = int(report_length)
    ngram_order = int(ngram_order)

    repetition_examples = repetition_utils.repetition_examples(ref=ref,
                                                               outs=outs,
                                                               src=src,
                                                               num_examples=report_length,
                                                               adjacent=adjacent,
                                                               ngram_order=ngram_order,
                                                               ignore_legitimate_reps=ignore_legitimate_reps)

    reporter = reporters.RepetitionExamplesReport(ref=ref,
                                                  outs=outs,
                                                  src=src,
                                                  repetition_examples=repetition_examples,
                                                  title=title,
                                                  report_length=report_length,
                                                  adjacent=adjacent,
                                                  ngram_order=ngram_order,
                                                  ignore_legitimate_reps=ignore_legitimate_reps)

    reporter.generate_report()

    return reporter


def generate_lang_id_report(ref, outs,
                            model="wtl",
                            min_length=5,
                            print_lines=False,
                            print_line_numbers=False):
    if model=="wtl":
        wtl = WhatTheLang()
    lang_id_reports=[]
    lang_id_lines_reports=[]
    lang_id_line_numbers_reports=[]
    for out in outs:
        langs = defaultdict(int)
        lang_lines = defaultdict(list)
        lang_line_numbers = defaultdict(list)
        for i, sentence in enumerate(out, start=1):
            line = corpus_utils.list2str(sentence)
            if len(sentence) >= int(min_length):
                if model=="langid":
                    (lang, prob) = langid.classify(line)
                elif model=="wtl":
                    lang = wtl.predict_lang(line)
                else:
                    raise NotImplementedError(f"Unknown model for language identification: '{model}'.")
                langs[lang] +=1
                if print_line_numbers:
                    lang_line_numbers[lang].append(i)
                if print_lines:    
                    lang_lines[lang].append(line)
            else:
                langs["shorter than min_length"] +=1
                if print_line_numbers:
                    lang_line_numbers["shorter than min_length"].append(i)
                if print_lines:
                    lang_lines["shorter than min_length"].append(line)
        lang_id_reports.append(langs)  
        lang_id_lines_reports.append(lang_lines)
        lang_id_line_numbers_reports.append(lang_line_numbers)

    reporter = reporters.LangIDreport(model, lang_id_reports, lang_id_lines_reports, lang_id_line_numbers_reports,print_lines,print_line_numbers)
    reporter.generate_report()
    return reporter

def main():
  parser = argparse.ArgumentParser(
      description='Program to compare MT results',
  )
  parser.add_argument('ref_file', type=str,
                      help='A path to a correct reference file')
  parser.add_argument('out_files', type=str, nargs='+',
                      help='Paths to system outputs')
  parser.add_argument('--sys_names', type=str, nargs='+', default=None,
                      help='Names for each system, must be same number as output files')
  parser.add_argument('--src_file', type=str, default=None,
                      help='A path to the source file')
  parser.add_argument('--fig_size', type=str, default='6x4.5',
                      help='The size of figures, in "width x height" format.')
  parser.add_argument('--compare_scores', type=str, nargs='*',
                      default=['score_type=bleu', 'score_type=length'],
                      help="""
                      Compare scores. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_score_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_word_accuracies', type=str, nargs='*',
                      default=['bucket_type=freq'],
                      help="""
                      Compare word accuracies by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_src_word_accuracies', type=str, nargs='*',
                      default=None,
                      help="""
                      Source analysis. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_src_word_accuracy_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_buckets', type=str, nargs='*',
                      default=['bucket_type=length,statistic_type=score,score_measure=bleu',
                               'bucket_type=lengthdiff',
                               'bucket_type=score,score_measure=sentbleu'],
                      help="""
                      Compare sentence counts by buckets. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_sentence_buckets_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_ngrams', type=str, nargs='*',
                      default=['compare_type=match'],
                      help="""
                      Compare ngrams. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_ngram_report' to see which arguments are available.
                      """)
  parser.add_argument('--compare_sentence_examples', type=str, nargs='*',
                      default=['score_type=sentbleu'],
                      help="""
                      Compare sentences. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                      See documentation for 'generate_sentence_examples' to see which arguments are available.
                      """)
  parser.add_argument('--compare_repetitions', type=str, nargs='*',
                      default=None,
                      help="""
                        Compare repetition statistics. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                        See documentation for 'generate_repetitions_report' to see which arguments are available.
                        """)
  parser.add_argument('--compare_repetition_examples', type=str, nargs='*',
                      default=None,
                      help="""
                        Compare sentences that contain repetitions. Can specify arguments in 'arg1=val1,arg2=val2,...' format.
                        See documentation for 'generate_repetition_examples' to see which arguments are available.
                        """)

  parser.add_argument('--output_directory', type=str, default=None,
                      help="""
                      A path to a directory where a graphical report will be saved. Open index.html in the directory
                      to read the report.
                      """)
  parser.add_argument('--report_title', type=str, default='compare-mt Analysis Report',
                      help="""
                      The name of the HTML report.
                      """)
  parser.add_argument('--decimals', type=int, default=4,
                      help="Number of decimals to print for floating point numbers")
  parser.add_argument('--scorer_scale', type=float, default=100, choices=[1, 100],
                      help="Set the scale of BLEU, METEOR, WER and chrF to 0-1 or 0-100 (default 0-100)")
  parser.add_argument('--lang_id', type=str, nargs='*', default=None,
                      help="""
                      Use language identification on output. Can specify arguments in 'arg1=val1,arg2=val2,...' format. 
                      Arguments: model=[wtl,langid], min_length=int, print_lines=[True,False], print_line_numbers=[True,False]
                      Set minimum length for segments to be analyzed with language identification (the shorter the segment, the more unreliable the analysis), default=5.
                      """) 
  args = parser.parse_args()

  # Set formatting
  formatting.fmt.set_decimals(args.decimals)

  # Set scale
  scorers.global_scorer_scale = args.scorer_scale

  ref = corpus_utils.load_tokens(args.ref_file)
  outs = [corpus_utils.load_tokens(x) for x in args.out_files]

  src = corpus_utils.load_tokens(args.src_file) if args.src_file else None 
  reporters.sys_names = args.sys_names if args.sys_names else [f'sys{i+1}' for i in range(len(outs))]
  reporters.fig_size = tuple([float(x) for x in args.fig_size.split('x')])
  if len(reporters.sys_names) != len(outs):
    raise ValueError(f'len(sys_names) != len(outs) -- {len(reporters.sys_names)} != {len(outs)}')

  reports = []

  report_types = [
    (args.compare_scores, generate_score_report, 'Aggregate Scores', False),
    (args.compare_word_accuracies, generate_word_accuracy_report, 'Word Accuracies', False),
    (args.compare_src_word_accuracies, generate_src_word_accuracy_report, 'Source Word Accuracies', True),
    (args.compare_sentence_buckets, generate_sentence_bucketed_report, 'Sentence Buckets', False),
    (args.compare_repetitions, generate_repetitions_report, 'Repetition Statistics', True),
    (args.compare_repetition_examples, generate_repetitions_examples, 'Repetition Examples', True),
    (args.lang_id, generate_lang_id_report, 'Language Identification', False)]
  if len(outs) > 1:
    report_types += [
      (args.compare_ngrams, generate_ngram_report, 'Characteristic N-grams', False),
      (args.compare_sentence_examples, generate_sentence_examples, 'Sentence Examples', True),
    ]

  for arg, func, name, use_src in report_types:
    if arg is not None:
      if use_src:
        reports.append( (name, [func(ref, outs, src, **arg_utils.parse_profile(x)) for x in arg]) )
      else:
        reports.append( (name, [func(ref, outs, **arg_utils.parse_profile(x)) for x in arg]) )

  # Write all reports into a single html file
  if args.output_directory != None:
    reporters.generate_html_report(reports, args.output_directory, args.report_title)

if __name__ == '__main__':
  main()
