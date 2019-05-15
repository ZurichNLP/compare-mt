import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
import numpy as np
import os
from compare_mt.formatting import fmt

# Global variables used by all reporters. These are set by compare_mt_main.py
sys_names = None
fig_size = None

# The CSS style file to use
css_style = """
html {
  font-family: sans-serif;
}

table, th, td {
  border: 1px solid black;
}

th, td {
  padding: 2px;
}

tr:hover {background-color: #f5f5f5;}

tr:nth-child(even) {background-color: #f2f2f2;}

th {
  background-color: #396AB1;
  color: white;
}

caption {
  font-size: 14pt;
  font-weight: bold;
}

table {
  border-collapse: collapse;
}
"""

# The Javascript header to use
javascript_style = """
function showhide(elem) {
  var x = document.getElementById(elem);
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}
"""

fig_counter, tab_counter = 0, 0
def next_fig_id():
  global fig_counter
  fig_counter += 1
  return f'{fig_counter:03d}'
def next_tab_id():
  global tab_counter
  tab_counter += 1
  return f'{tab_counter:03d}'

bar_colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60", "#808585", "#9067A7", "#AB6857", "#CCC210"]

def make_bar_chart(datas,
                   output_directory, output_fig_file, output_fig_format='png',
                   errs=None, title=None, xlabel=None, xticklabels=None, ylabel=None):
  fig, ax = plt.subplots(figsize=fig_size)
  ind = np.arange(len(datas[0]))
  width = 0.7/len(datas)
  bars = []
  for i, data in enumerate(datas):
    err = errs[i] if errs != None else None
    bars.append(ax.bar(ind+i*width, data, width, color=bar_colors[i], bottom=0, yerr=err))
  # Set axis/title labels
  if title is not None:
    ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xticklabels is not None:
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=70)
  else:
    ax.xaxis.set_visible(False) 

  ax.legend(bars, sys_names)
  ax.autoscale_view()

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  out_file = os.path.join(output_directory, f'{output_fig_file}.{output_fig_format}')
  plt.savefig(out_file, format=output_fig_format, bbox_inches='tight')

def html_img_reference(fig_file, title):
  latex_code_pieces = [r"\begin{figure}[h]",
                       r"  \centering",
                       r"  \includegraphics{" + fig_file + ".pdf}",
                       r"  \caption{" + title + "}",
                       r"  \label{fig:" + fig_file + "}",
                       r"\end{figure}"]
  latex_code = "\n".join(latex_code_pieces)
  return (f'<img src="{fig_file}.png" alt="{title}"> <br/>' +
          f'<button onclick="showhide(\'{fig_file}_latex\')">Show/Hide LaTeX</button> <br/>' +
          f'<pre id="{fig_file}_latex" style="display:none">{latex_code}</pre>')

class Report: 
  # def __init__(self, iterable=(), **kwargs):
  #   # Initialize a report by a dictionary which contains all the statistics
  #   self.__dict__.update(iterable, **kwargs)
  
  def print(self): 
    raise NotImplementedError('print must be implemented in subclasses of Report')

  def plot(self, output_directory, output_fig_file, output_fig_type):
    raise NotImplementedError('plot must be implemented in subclasses of Report')

  def print_header(self, header):
    print(f'********************** {header} ************************')

  def print_tabbed_table(self, tab):
    for x in tab:
      print('\t'.join([fmt(y) if y else '' for y in x]))
    print()

  def generate_report(self, output_fig_file=None, output_fig_format=None, output_directory=None):
    self.print()

class ScoreReport(Report):
  def __init__(self, scorer, scores, strs,
               wins=None, sys_stats=None, prob_thresh=0.05,
               title=None):
    self.scorer = scorer 
    self.scores = scores
    self.strs = [f'{fmt(x)} ({y})' if y else fmt(x) for (x,y) in zip(scores,strs)]
    self.wins = wins
    self.sys_stats = sys_stats
    self.output_fig_file = f'{next_fig_id()}-score-{scorer.idstr()}'
    self.prob_thresh = prob_thresh
    self.title = scorer.name() if not title else title

  def winstr_pval(self, my_wins):
    if 1-my_wins[0] < self.prob_thresh:
      winstr = 's1>s2'
    elif 1-my_wins[1] < self.prob_thresh:
      winstr = 's2>s1'
    else:
      winstr = '-'
    pval = 1-(my_wins[0] if my_wins[0] > my_wins[1] else my_wins[1])
    return winstr, pval

  def scores_to_tables(self):
    if self.wins is None:
      # Single table with just scores
      return [[""]+sys_names, [self.scorer.name()]+self.strs], None
    elif len(self.scores) == 1:
      # Single table with scores for one system
      return [
        [""]+sys_names,
        [self.scorer.name()]+self.strs,
        [""]+[f'[{x["lower_bound"]:.4f},{x["upper_bound"]:.4f}]' for x in self.sys_stats]
      ], None
    elif len(self.scores) == 2:
      # Single table with scores and wins for two systems
      winstr, pval = self.winstr_pval(self.wins[0][1])
      return [
        [""]+sys_names+["Win?"],
        [self.scorer.name()]+self.strs+[winstr],
        [""]+[f'[{fmt(x["lower_bound"])},{fmt(x["upper_bound"])}]' for x in self.sys_stats]+[f'p={fmt(pval)}']
      ], None
    else:
      # Table with scores, and separate one with wins for multiple systems
      wptable = [['v s1 / s2 ->'] + [sys_names[i] for i in range(1,len(self.scores))]]
      for i in range(0, len(self.scores)-1):
        wptable.append([sys_names[i]] + [""] * (len(self.scores)-1))
      for (left,right), my_wins in self.wins:
        winstr, pval = self.winstr_pval(my_wins)
        wptable[left+1][right] = f'{winstr} (p={fmt(pval)})'
      return [[""]+sys_names, [self.scorer.name()]+self.strs], wptable

  def print(self):
    aggregate_table, win_table = self.scores_to_tables()
    self.print_header('Aggregate Scores')
    print(f'{self.title}:')
    self.print_tabbed_table(aggregate_table)
    if win_table:
      self.print_tabbed_table(win_table)

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    sys = [[score] for score in self.scores]
    if self.wins:
      sys_errs = [np.array([[score-stat['lower_bound'], stat['upper_bound']-score]]) for (score,stat) in zip(self.scores, self.sys_stats)]
    else:
      sys_errs = None
    xticklabels = None

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   errs=sys_errs, ylabel=self.scorer.name(),
                   xticklabels=xticklabels)

  def html_content(self, output_directory):
    aggregate_table, win_table = self.scores_to_tables()
    html = html_table(aggregate_table, title=self.title)
    if win_table:
      html += html_table(win_table, title=f'{self.scorer.name()} Wins')
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Score Comparison')
    return html
    
class WordReport(Report):
  def __init__(self, bucketer, matches, acc_type, header, title=None):
    self.bucketer = bucketer
    self.matches = [[m for m in match] for match in matches]
    self.acc_type = acc_type
    self.header = header
    self.acc_type_map = {'prec': 3, 'rec': 4, 'fmeas': 5}
    self.output_fig_file = f'{next_fig_id()}-wordacc-{bucketer.name()}'
    self.title = title if title else f'word {acc_type} by {bucketer.name()} bucket'

  def print(self):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.matches, self.acc_type, self.header
    self.print_header(header)
    acc_types = acc_type.split('+')
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      print(f'--- {self.title}')
      for i, bucket_str in enumerate(bucketer.bucket_strs):
        print(f'{bucket_str}', end='')
        for match in matches:
          print(f'\t{fmt(match[i][aid])}', end='')
        print()
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    acc_types = self.acc_type.split('+')
    for at in acc_types:
      if at not in self.acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = self.acc_type_map[at]
      sys = [[m[aid] for m in match] for match in self.matches]
      xticklabels = [s for s in self.bucketer.bucket_strs] 

      make_bar_chart(sys,
                     output_directory, output_fig_file,
                     output_fig_format=output_fig_format,
                     xlabel=self.bucketer.name(), ylabel=at,
                     xticklabels=xticklabels)
    
  def html_content(self, output_directory):
    acc_type_map = self.acc_type_map
    bucketer, matches, acc_type, header = self.bucketer, self.matches, self.acc_type, self.header
    acc_types = acc_type.split('+') 

    html = ''
    for at in acc_types:
      if at not in acc_type_map:
        raise ValueError(f'Unknown accuracy type {at}')
      aid = acc_type_map[at]
      title = f'Word {acc_type} by {bucketer.name()} bucket' if not self.title else self.title
      table = [[bucketer.name()] + sys_names]
      for i, bs in enumerate(bucketer.bucket_strs):
        line = [bs]
        for match in matches:
          line.append(f'{fmt(match[i][aid])}')
        table += [line] 
      html += html_table(table, title)
      img_name = f'{self.output_fig_file}-{at}'
      for ext in ('png', 'pdf'):
        self.plot(output_directory, img_name, ext)
      html += html_img_reference(img_name, self.header)
    return html 

class NgramReport(Report):
  def __init__(self, scorelist, report_length, min_ngram_length, max_ngram_length,
               matches, compare_type, alpha, compare_directions=[(0, 1)], label_files=None, title=None):
    self.scorelist = scorelist
    self.report_length = report_length 
    self.min_ngram_length = min_ngram_length
    self.max_ngram_length = max_ngram_length
    self.matches = matches
    self.compare_type = compare_type
    self.label_files = label_files
    self.alpha = alpha
    self.compare_directions = compare_directions
    self.title = title

  def print(self):
    report_length = self.report_length
    self.print_header('N-gram Difference Analysis')
    if self.title:
      print(f'--- {self.title}')
    else:
      print(f'--- min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
      print(f'    report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')

    if self.label_files is not None:
      print(self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      print(f'--- {report_length} n-grams where {sys_names[left]}>{sys_names[right]} in {self.compare_type}')
      for k, v in self.scorelist[i][:report_length]:
        print(f"{' '.join(k)}\t{fmt(v)} (sys{left+1}={self.matches[left][k]}, sys{right+1}={self.matches[right][k]})")
      print()
      print(f'--- {report_length} n-grams where {sys_names[right]}>{sys_names[left]} in {self.compare_type}')
      for k, v in reversed(self.scorelist[i][-report_length:]):
        print(f"{' '.join(k)}\t{fmt(v)} (sys{left+1}={self.matches[left][k]}, sys{right+1}={self.matches[right][k]})")
      print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    raise NotImplementedError('Plotting is not implemented for n-gram reports')

  def html_content(self, output_directory=None):
    report_length = self.report_length
    if self.title:
      html = tag_str('p', self.title)
    else:
      html = tag_str('p', f'min_ngram_length={self.min_ngram_length}, max_ngram_length={self.max_ngram_length}')
      html += tag_str('p', f'report_length={report_length}, alpha={self.alpha}, compare_type={self.compare_type}')
      if self.label_files is not None:
        html += tag_str('p', self.label_files)

    for i, (left, right) in enumerate(self.compare_directions):
      title = f'{report_length} n-grams where {sys_names[left]}>{sys_names[right]} in {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), fmt(v), self.matches[left][k], self.matches[right][k]] for k, v in self.scorelist[i][:report_length]])
      html += html_table(table, title)

      title = f'{report_length} n-grams where {sys_names[right]}>{sys_names[left]} in {self.compare_type}'
      table = [['n-gram', self.compare_type, f'{sys_names[left]}', f'{sys_names[right]}']]
      table.extend([[' '.join(k), fmt(v), self.matches[left][k], self.matches[right][k]] for k, v in reversed(self.scorelist[i][-report_length:])])
      html += html_table(table, title)
    return html 

class SentenceReport(Report):

  def __init__(self, bucketer=None, sys_stats=None, statistic_type=None, scorer=None, title=None):
    self.bucketer = bucketer
    self.sys_stats = [[s for s in stat] for stat in sys_stats]
    self.statistic_type = statistic_type
    self.scorer = scorer
    self.yname = scorer.name() if statistic_type == 'score' else statistic_type
    self.yidstr = scorer.idstr() if statistic_type == 'score' else statistic_type
    self.output_fig_file = f'{next_fig_id()}-sent-{bucketer.idstr()}-{self.yidstr}'
    if title:
      self.title = title
    elif scorer:
      self.title = f'bucket type: {bucketer.name()}, statistic type: {scorer.name()}'
    else:
      self.title = f'bucket type: {bucketer.name()}, statistic type: {statistic_type}'

  def print(self):
    self.print_header('Sentence Bucket Analysis')
    print(f'--- {self.title}')
    for i, bs in enumerate(self.bucketer.bucket_strs):
      print(f'{bs}', end='')
      for stat in self.sys_stats:
        print(f'\t{fmt(stat[i])}', end='')
      print()
    print()

  def plot(self, output_directory='outputs', output_fig_file='word-acc', output_fig_format='pdf'):
    sys = self.sys_stats
    xticklabels = [s for s in self.bucketer.bucket_strs] 

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   xlabel=self.bucketer.name(), ylabel=self.yname,
                   xticklabels=xticklabels)

  def html_content(self, output_directory=None):
    table = [ [self.bucketer.idstr()] + sys_names ]
    for i, bs in enumerate(self.bucketer.bucket_strs):
      line = [bs]
      for stat in self.sys_stats:
        line.append(fmt(stat[i]))
      table.extend([line])
    html = html_table(table, self.title)
    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Sentence Bucket Analysis')
    return html 

class SentenceExampleReport(Report):

  def __init__(self, report_length=None, scorediff_lists=None, scorer=None, ref=None, outs=None, src=None, compare_directions=[(0, 1)], title=None):
    self.report_length = report_length 
    self.scorediff_lists = scorediff_lists
    self.scorer = scorer
    self.ref = ref
    self.outs = outs
    self.src = src
    self.compare_directions = compare_directions
    self.title = title

  def print(self):
    self.print_header('Sentence Examples Analysis')
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      sleft, sright = sys_names[left], sys_names[right]
      print(f'--- {report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        print(f"{sleft}-{sright}={fmt(-bdiff)}, {sleft}={fmt(s1)}, {sright}={fmt(s2)}")
        if self.src:
          print(f"Src:  {' '.join(self.src[i])}")
        print ( 
          f"Ref:  {' '.join(ref[i])}\n"
          f"{sleft}: {' '.join(out1[i])}\n"
          f"{sright}: {' '.join(out2[i])}\n"
        )

      print(f'--- {report_length} sentences where {sright}>{sleft} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        print(f"{sleft}-{sright}={fmt(-bdiff)}, {sleft}={fmt(s1)}, {sright}={fmt(s2)}")
        if self.src:
          print(f"Src:  {' '.join(self.src[i])}")
        print (
          f"Ref:  {' '.join(ref[i])}\n"
          f"{sleft}: {' '.join(out1[i])}\n"
          f"{sright}: {' '.join(out2[i])}\n"
        )

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    pass 

  def html_content(self, output_directory=None):
    report_length = self.report_length
    for cnt, (left, right) in enumerate(self.compare_directions):
      sleft, sright = sys_names[left], sys_names[right]
      ref, out1, out2 = self.ref, self.outs[left], self.outs[right]
      html = tag_str('h4', f'{report_length} sentences where {sleft}>{sright} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][:report_length]:
        table = [['', 'Output', f'{self.scorer.idstr()}']]
        if self.src:
          table.append(['Src', ' '.join(self.src[i]), ''])
        table += [
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), fmt(s1)],
          [f'{sright}', ' '.join(out2[i]), fmt(s2)]
        ]
        
        html += html_table(table, None)

      html += tag_str('h4', f'{report_length} sentences where {sright}>{sleft} at {self.scorer.name()}')
      for bdiff, s1, s2, str1, str2, i in self.scorediff_lists[cnt][-report_length:]:
        table = [['', 'Output', f'{self.scorer.idstr()}']]
        if self.src:
          table.append(['Src', ' '.join(self.src[i]), ''])
        table += [
          ['Ref', ' '.join(ref[i]), ''],
          [f'{sleft}', ' '.join(out1[i]), fmt(s1)],
          [f'{sright}', ' '.join(out2[i]), fmt(s2)]
        ]

        html += html_table(table, None)

    return html

class LangIDreport(Report):
    
    def __init__(self, model, lang_id_reports, lang_id_lines_reports, lang_id_line_numbers_reports, print_lines, print_line_numbers):
        self.model= model
        self.lang_id_reports = lang_id_reports
        self.lang_id_lines_reports = lang_id_lines_reports
        self.lang_id_line_numbers_reports = lang_id_line_numbers_reports
        self.print_lines = print_lines
        self.print_line_numbers = print_line_numbers
        
        
    def print(self):
        self.print_header('Language Identification Analysis with '+ str(self.model) )
        for i, (lang_id_report, lang_id_lines_report, lang_id_line_numbers_report) in enumerate(zip(self.lang_id_reports, self.lang_id_lines_reports, self.lang_id_line_numbers_reports)):
            total_sents = sum(lang_id_report.values())
            sorted_langs = sorted(lang_id_report.items(), key=lambda v: v[1]) # sorted_langs = sorted list of tuples (lang, frequency)
            # print percentage of languages
            print("Output {}:".format(i))
            for lang, v in sorted_langs:
                print("{}: {}%".format(lang, float(v/total_sents *100)))
            # print lines classified as minority languages
            if self.print_lines:
                print()
                print("\t Language Lines:")
                for j in range(0,len(sorted_langs)):
                    (lang, freq) = sorted_langs[j]
                    if lang != "shorter than min_length":
                        print("{}: {}".format(lang, lang_id_lines_report[lang]))
                        
            if self.print_line_numbers:
                print()
                print("\t Language Line Numbers:")
                for j in range(0,len(sorted_langs)):
                    (lang, freq) = sorted_langs[j]
                    if lang != "shorter than min_length":
                        print("{}: {}".format(lang, lang_id_line_numbers_report[lang]))


class RepetitionReport(Report):

  def __init__(self, ref=None, outs=None, src=None, title=None, repetition_stats=None,
               adjacent=True, ngram_order=1, subtract_legitimate_reps=False):
    self.ref = ref
    self.outs = outs
    self.src = src
    self.repetition_stats = repetition_stats
    self.adjacent = adjacent
    self.ngram_order = ngram_order
    self.subtract_legitimate_reps = subtract_legitimate_reps

    self.output_fig_file = f'{next_fig_id()}-total-reps'

    if title:
      self.title = title
    else:
      self.title = f'adjacent={adjacent}, ngram_order={ngram_order}, subtract_legitimate_reps={subtract_legitimate_reps}'

  def print(self):
    self.print_header('Repetition Statistics Analysis')
    print(f'--- {self.title}')
    print(f'\tTOTAL REPS IN CORPUS')
    for sys_name, reps_total in zip(sys_names, self.repetition_stats):
      print(f'{sys_name}\t{reps_total}')
    print()

  def plot(self, output_directory='outputs', output_fig_file='rep-stats', output_fig_format='pdf'):
    sys = [[score] for score in self.repetition_stats]

    make_bar_chart(sys,
                   output_directory, output_fig_file,
                   output_fig_format=output_fig_format,
                   xlabel="", ylabel="TOTAL_REPS_IN_CORPUS",
                   xticklabels=None)

  def html_content(self, output_directory=None):

    html = tag_str('p', self.title)

    table = [['', 'TOTAL_REPS_IN_CORPUS']]

    for sys_name, reps_total in zip(sys_names, self.repetition_stats):

      table.append([f'{sys_name}', f'{reps_total}'])

    html += html_table(table, None)

    for ext in ('png', 'pdf'):
      self.plot(output_directory, self.output_fig_file, ext)
    html += html_img_reference(self.output_fig_file, 'Repetition Statistics Analysis')

    return html


class RepetitionExamplesReport(Report):

  def __init__(self, ref=None, outs=None, src=None, title=None, report_length=10, repetition_examples=None,
               adjacent=True, ngram_order=1, ignore_legitimate_reps=False):
    self.ref = ref
    self.outs = outs
    self.src = src
    self.title = title
    self.report_length = report_length
    self.repetition_examples = repetition_examples

    if title:
      self.title = title
    else:
      self.title = f'adjacent={adjacent}, ngram_order={ngram_order}, ignore_legitimate_reps={ignore_legitimate_reps}, report_length={report_length}'

  def print(self):
    self.print_header('Repetition Examples Analysis')
    print(f'--- {self.title}')

    for sys_name, examples, out in zip(sys_names, self.repetition_examples, self.outs):
      print()
      print(f'--- {self.report_length} worst examples from {sys_name}')
      for index in examples:
        if self.src:
          print(f"Src:  {' '.join(self.src[index])}")
        print(f"Ref:  {' '.join(self.ref[index])}")
        print(f"{sys_name}: {' '.join(out[index])}")
        print()
    print()

  def plot(self, output_directory, output_fig_file, output_fig_format='pdf'):
    pass

  def html_content(self, output_directory=None):

    html = tag_str('p', self.title)

    for sys_name, examples, out in zip(sys_names, self.repetition_examples, self.outs):
      html += tag_str('h4', f'{self.report_length} worst examples from {sys_name}')

      for index in examples:
        table = [['', 'Output']]

        if self.src:
          table.append(['Src', ' '.join(self.src[index])])
        table.append(['Ref', ' '.join(self.ref[index])])
        table.append([f'{sys_name}', ' '.join(out[index])])

        html += html_table(table, None)

    return html


def tag_str(tag, str, new_line=''):
  return f'<{tag}>{new_line} {str} {new_line}</{tag}>'

def html_table(table, title=None, bold_rows=1, bold_cols=1):
  html = '<table border="1">\n'
  if title is not None:
    html += tag_str('caption', title)
  for i, row in enumerate(table):
    tag_type = 'th' if (i < bold_rows) else 'td'
    table_row = '\n  '.join(tag_str('th' if j < bold_cols else tag_type, rdata) for (j, rdata) in enumerate(row))
    html += tag_str('tr', table_row)
  html += '\n</table>\n <br/>'

  tab_id = next_tab_id()
  latex_code = "\\begin{table}[t]\n  \\centering\n"
  cs = ['c'] * len(table[0])
  if bold_cols != 0:
    cs[bold_cols-1] = 'c||'
  latex_code += "  \\begin{tabular}{"+''.join(cs)+"}\n"
  for i, row in enumerate(table):
    latex_code += ' & '.join([fmt(x) for x in row]) + (' \\\\\n' if i != bold_rows-1 else ' \\\\ \\hline \\hline\n')
  latex_code += "  \\end{tabular}\n  \\caption{Caption}\n  \\label{tab:table"+tab_id+"}\n\\end{table}"

  html += (f'<button onclick="showhide(\'{tab_id}_latex\')">Show/Hide LaTeX</button> <br/>' +
           f'<pre id="{tab_id}_latex" style="display:none">{latex_code}</pre>')
  return html

def generate_html_report(reports, output_directory, report_title):
  content = []
  for name, rep in reports:
    content.append(f'<h2>{name}</h2>')
    for r in rep:
      content.append(r.html_content(output_directory))
  content = "\n".join(content)
  
  if not os.path.exists(output_directory):
        os.makedirs(output_directory)
  html_file = os.path.join(output_directory, 'index.html')
  with open(html_file, 'w') as f:
    content = content.encode("ascii","xmlcharrefreplace").decode()
    message = (f'<html>\n<head>\n<link rel="stylesheet" href="compare_mt.css">\n</head>\n'+
               f'<script>\n{javascript_style}\n</script>\n'+
               f'<body>\n<h1>{report_title}</h1>\n {content} \n</body>\n</html>')
    f.write(message)
  css_file = os.path.join(output_directory, 'compare_mt.css')
  with open(css_file, 'w') as f:
    f.write(css_style)
  
