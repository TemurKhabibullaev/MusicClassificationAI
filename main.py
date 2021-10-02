# Temur Khabibullaev on Music Classification with AI project
import sys
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import scipy
import scipy.io.wavfile
import collections
import csv
import urllib.request
import tarfile

CHART_DIR = "charts"
if not Path(CHART_DIR).exists():
    os.mkdir(CHART_DIR)

DATA_DIR = "data"
if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

GENRE_DIR = Path(DATA_DIR) / 'genres'
plt.style.use('seaborn')
DPI = 100


def save_png(name):
    fn = 'B09124_11_%s.png' % name  # please ignore, it just helps our publisher :-)
    plt.savefig(str(Path(CHART_DIR) / fn), bbox_inches="tight")


def plot_pr(auc_score, name, precision, recall, label=None, plot_nr=None):
    plt.figure(num=None, figsize=(5, 4), dpi=DPI)
    plt.grid(True)
    plt.fill_between(recall, precision, alpha=0.5)
    plt.plot(recall, precision, lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P/R curve (AUC=%0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    save_png("%s_pr_%s" % (plot_nr, filename))


def plot_roc(auc_score, name, tpr, fpr, label=None, plot_nr=None):
    plt.figure(num=None, figsize=(5, 4), dpi=DPI)
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.fill_between(fpr, tpr, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (AUC = %0.2f) / %s' % (auc_score, label), verticalalignment="bottom")
    plt.legend(loc="lower right")
    save_png('%i_auc_%s' % (plot_nr, name))


genre_fn = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
urllib.request.urlretrieve(genre_fn, Path(DATA_DIR) / 'genres.tar.gz')