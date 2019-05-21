# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import os
from os import listdir
from os.path import isfile, join
import shutil

import matplotlib.pyplot as plt
import numpy as np

def buildTimingLegendEntry(tableName, durations):
    return "{} (min: {:.2f}, max: {:.2f}, mean: {:.2f})".format(table.split('_')[-1], min(durations), max(durations), np.mean(durations))

def dist(gt, rec):
    return ((gt[0] - rec[0])**2 + (gt[1] - rec[1])**2)**0.5


experimentFolder = 'C:/TEMP/SDD_Experiments/out'
outFolder = 'C:/TEMP/SDD_Experiments/out/figures'

if not os.path.exists(outFolder):
    os.makedirs(outFolder)
   
dbFiles = [f for f in listdir(experimentFolder) if isfile(join(experimentFolder, f))]

timingTables = ['Timing_OptimizationProblem::Solve', 'Timing_OptimizationProblem::SolveCplex']

# Make DataFrame
for dbFile in dbFiles:
    for table in timingTables:

        connection = sqlite3.connect(join(experimentFolder, dbFile))
        c = connection.cursor()

        rows = [x for x in c.execute("SELECT * FROM '" + table + "';")]
        frames, timingId, start, stop, duration = zip(*rows)
        if any([t != 0 for t in timingId]):
            raise RuntimeError("I don't want to handle this right now")

       

        plt.clf()
        plt.cla()

        plt.plot(frames, duration, label=buildTimingLegendEntry(table, duration))
    
    plt.ylim([0, 600])
    plt.xlabel('Frame')
    plt.ylabel('Duration (ms)')

    plt.title(dbFile)

    #plt.legend(bbox_to_anchor=(0., 1.07, 1., .102), loc=3, ncol=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(join(outFolder, "timings_" + dbFile[:-3] + ".png"), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.cla()

    rows_gt = [x for x in c.execute("SELECT * FROM 'GroundTruth';")]
    rows_rec = [x for x in c.execute("SELECT * FROM 'FusedSolution';")]

    frames_gt, actors_gt, pos_x_gt, pos_y_gt, pos_z_gt, size_x_gt, size_y_gt, size_z_gt = zip(*rows_gt)
    frames_rec, actors_rec, pos_x_rec, pos_y_rec, pos_z_rec, size_x_rec, size_y_rec, size_z_rec = zip(*rows_rec)

    gt = { }
    for i, f in enumerate(frames_gt):
        if f not in gt:
            gt[f] = []
        gt[f].append((pos_x_gt[i], pos_y_gt[i]))

    rec = { }
    for i, f in enumerate(frames_rec):
        if f not in rec:
            rec[f] = []
        rec[f].append((pos_x_rec[i], pos_y_rec[i]))

    allFrames = list(set(gt.keys()) | set(rec.keys()))
    distances = [0] * len(allFrames)
    numGt = [0] * len(allFrames)
    numRec = [0] * len(allFrames)

    for i, f in enumerate(allFrames):
        if f in gt.keys():
            numGt[i] += len(gt[f])
        if f in rec.keys():
            numRec[i] += len(rec[f])

        if f not in gt.keys() or f not in rec.keys():
            continue
        distances[i] = min([dist(gt[f][0], x) for x in rec[f]])

    fig, ax1 = plt.subplots()

    ax1.fill_between(allFrames, numRec, color='#c6bfa1')
    ax1.set_ylim([0, 5])
    ax1.set_ylabel('Num Objects')

    ax2 = ax1.twinx();
    ax2.plot(allFrames, distances, linewidth=0.2)
    ax2.set_ylim([0, 200])
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Distance (cm)')

    plt.title(dbFile)

    plt.savefig(join(outFolder, "distances_" + dbFile[:-3] + ".png"), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.cla()

    connection.close()

