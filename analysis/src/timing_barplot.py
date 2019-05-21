# -*- coding: utf-8 -*-

import sqlite3
import os
from os import listdir
from os.path import isfile, join
import shutil

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class SeqData:
    def __init__(self):
        self.frames = []
        self.cplex_durations = []


def buildTimingLegendEntry(tableName, durations):
    return "{} (min: {:.2f}, max: {:.2f}, mean: {:.2f})".format(table.split('_')[-1], min(durations), max(durations), np.mean(durations))

def dist(gt, rec):
    return ((gt[0] - rec[0])**2 + (gt[1] - rec[1])**2)**0.5

def autolabel(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = int(rect.get_height())
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')

def create_timingBarplot(experimentFolder, outFolder, timingTable, title):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
   
    dbFiles = [f for f in listdir(experimentFolder) if isfile(join(experimentFolder, f))]

    allAdvInits = set()
    allRootAlgs = set()
    seqData = dict()

    matplotlib.rcParams.update({'errorbar.capsize': 10})
    matplotlib.rc('font', **{'family': 'serif'})
    matplotlib.rcParams['text.usetex'] = True

    for dbFile in sorted(dbFiles):
        connection = sqlite3.connect(join(experimentFolder, dbFile))
        c = connection.cursor()

        configuration = [x for x in c.execute("SELECT `optimization.advancedInitialization`, `optimization.rootAlg` FROM 'Configuration';")]
        if len(configuration) > 1:
            raise RuntimeError("Configuration should only have one row.")

        currAdvInit = configuration[0][0]
        currRootAlg = configuration[0][1]

        allAdvInits.add(currAdvInit)
        allRootAlgs.add(currRootAlg)

        if not currAdvInit in seqData:
            seqData[currAdvInit] = dict()
        if not currRootAlg in seqData[currAdvInit]:
            seqData[currAdvInit][currRootAlg] = SeqData()

        rows = [x for x in c.execute("SELECT `ID_FRAME`, `duration` FROM '" + timingTable + "';")]
        seqData[currAdvInit][currRootAlg].frames, seqData[currAdvInit][currRootAlg].cplex_durations = zip(*rows)

        connection.close()

    # # # Plotting
    plt.style.use('seaborn-paper')
    
    fig, ax = plt.subplots()
    # plt.ylim([0, 600])
    plt.ylabel('Duration (ms)')

    rootAlg2realWord = dict( zip(allRootAlgs, allRootAlgs) )

    #plt.title(title)
    #plt.xscale(0.5)
    width = 0.4  # the width of the bars
    ind = np.arange( 2 ) # denn es gibt advInit oder nicht advInit

    rects = dict()
    ax.yaxis.grid(True, zorder=0)
    colors = ('#2e6284', '#357198', '#418dbf', '#66a4cd', '#8bbbda')

    for idx, (rootAlg, currColor) in enumerate(zip(sorted(allRootAlgs), colors)):
        means_rootAlg = ( np.mean( seqData[0][rootAlg].cplex_durations ), np.mean( seqData[1][rootAlg].cplex_durations ) )
        stds_rootAlg = ( np.std( seqData[0][rootAlg].cplex_durations ), np.std( seqData[1][rootAlg].cplex_durations ) )
   
        x = ind - width/2 + width/(len(allRootAlgs) - 1) * idx
        #rects[rootAlg] = ax.bar(x, means_rootAlg, width/len(allRootAlgs), yerr=stds_rootAlg, label=rootAlg, zorder=3, capsize=10, error_kw={'zorder':5})
        rects[rootAlg] = ax.bar(x, means_rootAlg, width/len(allRootAlgs), color=currColor, label=rootAlg, zorder=2)
        for j in range(len(x)):
            (_,caps,_) = ax.errorbar(x[j], means_rootAlg[j], stds_rootAlg[j], ecolor='black', capsize=3, linewidth=1, zorder=5)

            for cap in caps:
                cap.set_color('black')
                cap.set_markeredgewidth(1)

        

        #autolabel(ax, rects[rootAlg], "center")

    ax.set_xticks(ind)
    ax.set_xticklabels(('Zero Initialization', 'Advanced Initialization'))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    legend = ax.legend(loc='upper left', fancybox=False, framealpha=True);
    legend.get_frame().set_zorder(5);
    legend.get_frame().set_linewidth(1);
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')


    mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()
    
    fig.set_size_inches(5,3.5);
    #ax.subplots_adjust(right = 0.5)
    plt.savefig(join(outFolder, "timing_barplot.eps"), transparent=True, bbox_inches='tight', dpi=300)
   
    plt.show()


experimentFolder = r'C:\TEMP\SDD_Experiments\out_experiment_001'
outFolder = r'C:\TEMP\SDD_Experiments\out_experiment_001\figures'
table = r'Timing_OptimizationProblem::SolveCplex'
title = 'Experiment__SimpleForPrecision__(40_40_2.5)__(160_120)__((person_01))__(single_60)__(X_Y)__(1_ 30)__(0.5_100)'

create_timingBarplot(experimentFolder, outFolder, table, title)
# plt.clf()
# plt.cla()


