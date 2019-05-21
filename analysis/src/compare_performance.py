# -*- coding: utf-8 -*-

import sqlite3
import os
from os import listdir
from os.path import isfile, join
import shutil
from scipy.optimize import linear_sum_assignment 
import math

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

def getExpectationDist():
    sample_size = 1000000
    rnd_pnts = np.random.uniform(0., 0.5, size=(sample_size, 2))
    d = np.zeros(sample_size)

    for (idx, rnd_pnt) in enumerate(rnd_pnts):
        d[idx] = ((rnd_pnt[0])**2 + (rnd_pnt[1])**2)**0.5

    return np.mean(d)

def buildTimingLegendEntry(tableName, durations):
    return "{} (min: {:.2f}, max: {:.2f}, mean: {:.2f})".format(table.split('_')[-1], min(durations), max(durations), np.mean(durations))

def dist(gt, rec):
    return ((gt[0] - rec[0])**2 + (gt[1] - rec[1])**2)**0.5

def is_in_bounding_box( pos, bb_box ):
    return pos[0] >= bb_box[0] and pos[0] <= bb_box[2] and pos[1] >= bb_box[1] and pos[1] <= bb_box[3]


# experimentFolder = 
# outFolder = 
# experimentFolder = 
# outFolder = 
experimentFolders = [
    # r'\\datafs.ini.rub.de\rtcv\Experimente\SDD_SFS_Fusion (ITSC 2019)\out_experiment_001', 
    # r'\\datafs.ini.rub.de\rtcv\Experimente\SDD_SFS_Fusion (ITSC 2019)\out_experiment_002',
    # r'\\datafs.ini.rub.de\rtcv\Experimente\SDD_SFS_Fusion (ITSC 2019)\out_experiment_003'
        r'C:\TEMP\SDD_Experiments\out_experiment_001'
        ]
outFolders = [
    # r'd:\rtcv_results\ITSC_2019_MultiCamPedestrian\compare_performance_001',
    # r'd:\rtcv_results\ITSC_2019_MultiCamPedestrian\compare_performance_002',
    r'C:\TEMP\SDD_Experiments\out_experiment_001\figures'
        ]

for (experimentFolder, outFolder) in zip(experimentFolders, outFolders):

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    expectation_dist = getExpectationDist()
    
    dbFiles = [f for f in listdir(experimentFolder) if isfile(join(experimentFolder, f))]

    s = "Config; time_mean; time_min; time_max; distance_rec; distance_vol; distance_bb; precision; recall\n"


    for dbFile in dbFiles:
        connection = sqlite3.connect(join(experimentFolder, dbFile))
        c = connection.cursor()

        #configuration = [x for x in c.execute("SELECT `grid.distX`, `detector.maxMergingDistance`, `sfs.init.voxelSize` FROM 'Configuration';")]
        configuration = [x for x in c.execute("SELECT `grid.distX`, `optimization.rootAlg`, `optimization.advancedInitialization` FROM 'Configuration';")]
        if len(configuration) != 1:
            raise RuntimeError("Configuration should only have one row.")

        expected_error_min = float(configuration[0][0]) * expectation_dist

        conf_title = str(configuration[0][1]) + "_" + str(configuration[0][2]) 
        s += conf_title + "; "

        timings = [x for x in c.execute("SELECT `duration` from `Timing_Controller::ProcessInput`;")]
        s += str(np.mean(timings)) + "; " + str(np.min(timings)) + "; " + str(np.max(timings)) + "; "

        rows_gt = [x for x in c.execute("SELECT * FROM 'GroundTruth';")]
        rows_rec = [x for x in c.execute("SELECT * FROM 'FusedSolution';")]
        rows_vol = [x for x in c.execute("SELECT * FROM 'FusedVolume';")]

        try:
            frames_gt, actors_gt, pos_x_gt, pos_y_gt, pos_z_gt, size_x_gt, size_y_gt, size_z_gt = zip(*rows_gt)
            frames_rec, actors_rec, pos_x_rec, pos_y_rec, pos_z_rec, size_x_rec, size_y_rec, size_z_rec = zip(*rows_rec)
            frames_vol, actors_vol, cen_x, cen_y, cen_z, bb_x1, bb_y1, bb_z1, bb_x2, bb_y2, bb_z2 = zip(*rows_vol)
        except:
            break
        connection.close()

        frames_gt, actors_gt, pos_x_gt, pos_y_gt, pos_z_gt, size_x_gt, size_y_gt, size_z_gt = zip(*rows_gt)
        frames_rec, actors_rec, pos_x_rec, pos_y_rec, pos_z_rec, size_x_rec, size_y_rec, size_z_rec = zip(*rows_rec)
        frames_vol, actors_vol, cen_x, cen_y, cen_z, bb_x1, bb_y1, bb_z1, bb_x2, bb_y2, bb_z2 = zip(*rows_vol)

        gt = { } # gt positions for each frame
        for i, f in enumerate(frames_gt):
            if f not in gt:
                gt[f] = []
            gt[f].append((pos_x_gt[i], pos_y_gt[i]))

        rec = { } # 'FusedSolution' positions for each frame
        for i, f in enumerate(frames_rec):
            if f not in rec:
                rec[f] = []
            rec[f].append((pos_x_rec[i], pos_y_rec[i]))

        vol = { } # 'FusedVolume' positions for each frame
        bb = { } # 'FusedVolume' BoundingBox center positions for each frame
        bb_box = { } # 'FusedVolume' BoundingBox low-high_xy for each frame
        for i, f in enumerate(frames_vol):
            if f not in vol:
                vol[f] = []
            if f not in bb:
                bb[f] = []
            if f not in bb_box:
                bb_box[f] = []
            vol[f].append((cen_x[i], cen_y[i]))
            bb_cen_x = (bb_x1[i] + bb_x2[i]) / 2
            bb_cen_y = (bb_y1[i] + bb_y2[i]) / 2
            bb[f].append((bb_cen_x, bb_cen_y))
            bb_box[f].append((bb_x1[i], bb_y1[i], bb_x2[i], bb_y2[i]))


        allFrames = list(set(gt.keys()) | set(rec.keys()))
        numGt = [0] * max(allFrames)
        numRec = [0] * max(allFrames)
        numTP = [0] * max(allFrames)
        numFP = [0] * max(allFrames)
        numFN = [0] * max(allFrames)
        distances_rec = dict()
        distances_vol = dict()
        distances_bb = dict()

        gt_idxs = dict()
        bb_idxs = dict()

        for i, f in enumerate(allFrames):
            if f in gt.keys():
                numGt[i] += len(gt[f])
            if f in rec.keys():
                numRec[i] += len(rec[f])



            if f in gt.keys() and f in rec.keys():
                if not f in distances_rec.keys():
                    distances_rec[f] = []
                if not f in distances_vol.keys():
                    distances_vol[f] = []
                if not f in distances_bb.keys():
                    distances_bb[f] = []                        

                distMat = np.ones( [len(gt[f]), len(bb_box[f])] )
                for (i, curr_gt) in enumerate(gt[f]):
                    for (j, curr_bb_box) in enumerate(bb_box[f]):
                        distMat[i, j] = float(not is_in_bounding_box(curr_gt, curr_bb_box))

                gt_idxs, bb_idxs = linear_sum_assignment( distMat )
                for (i, curr_gt_pos) in enumerate(gt[f]):
                    if i in gt_idxs:
                        curr_bb_match_idx = int(bb_idxs[i == gt_idxs])
                        if distMat[i, curr_bb_match_idx] == 0:
                            numTP[f] += 1
                            # TP gefunden, Distanzen merken

                            distances_rec[f].append( dist(curr_gt_pos, rec[f][curr_bb_match_idx] ) )
                            distances_vol[f].append( dist(curr_gt_pos, vol[f][curr_bb_match_idx] ) )
                            distances_bb[f].append( dist(curr_gt_pos, bb[f][curr_bb_match_idx] ) )
                        else:
                            distances_rec[f].append( float('NaN') ) 
                            distances_vol[f].append( float('NaN') )
                            distances_bb[f].append( float('NaN') )
                            numFN[f] += 1
                    else:
                        distances_rec[f].append( float('NaN') ) 
                        distances_vol[f].append( float('NaN') )
                        distances_bb[f].append( float('NaN') )                        
                        numFN[f] += 1
                
                for (i, _) in enumerate(bb_box[f]):
                    if i in bb_idxs:
                        if distMat[gt_idxs[i == bb_idxs], i] != 0:
                            numFP[f] += 1                        
                    else:
                        numFP[f] += 1                        

            elif f in gt.keys():
                numFN[i] = len(gt[f])
            elif f in rec.keys():
                numFP[i] = len(rec[f])

        s += str(  np.mean(sum(distances_rec.values(), [])) ) + "; " + str(np.mean(sum(distances_vol.values(), []))) + "; " + str(np.mean(sum(distances_bb.values(), []))) + "; "

        totalTP = np.sum(numTP)
        totalFP = np.sum(numFP)
        totalFN = np.sum(numFN)

        s += str( totalTP / (totalFP + totalTP)  ) + "; " + str( totalTP / (totalFN + totalTP) ) + ";\n"

        fig = plt.figure()
        plt.title(conf_title)
        err_nonnan_rec = [x for x in sum(distances_rec.values(), []) if not math.isnan(x) ]
        plt.hist( err_nonnan_rec )
        hlabel_exp_min = plt.axvline(expected_error_min, linestyle='--', color='k')
        hlabel_mean = plt.axvline(np.mean(err_nonnan_rec), linestyle='--', color='b')
        plt.xlabel('Localizing Accuracy [cm]')
        plt.legend( [hlabel_exp_min, hlabel_mean], ['Expected error by grid discretization', 'Average localization error'])

        plt.savefig( join(outFolder, conf_title + '_dist_rec.png')  )

        fig = plt.figure()
        plt.title(conf_title)
        err_nonnan_vol = [x for x in sum(distances_vol.values(), []) if not math.isnan(x) ]
        plt.hist( err_nonnan_vol )
        hlabel_exp_min = plt.axvline(expected_error_min, linestyle='--', color='k')
        hlabel_mean = plt.axvline(np.mean(err_nonnan_vol), linestyle='--', color='b')
        plt.xlabel('Localizing Accuracy [cm]')
        plt.legend( [hlabel_exp_min, hlabel_mean], ['Expected error by grid discretization', 'Average localization error'])

        plt.savefig( join(outFolder, conf_title + '_dist_vol.png')  )

        fig = plt.figure()
        plt.title(conf_title)
        err_nonnan_bb = [x for x in sum(distances_bb.values(), []) if not math.isnan(x) ]
        plt.hist( err_nonnan_bb )
        hlabel_exp_min = plt.axvline(expected_error_min, linestyle='--', color='k')
        hlabel_mean = plt.axvline(np.mean(err_nonnan_bb), linestyle='--', color='b')
        plt.xlabel('Localizing Accuracy [cm]')
        plt.legend( [hlabel_exp_min, hlabel_mean], ['Expected error by grid discretization', 'Average localization error'])

        plt.savefig( join(outFolder, conf_title + '_dist_bb.png')  )


    


        gt_traj_x = []
        gt_traj_y = []
        err_traj_rec = []
        err_traj_vol = []
        err_traj_bb = []

        for k in range(10):
            try: # continue as long as there is data
                curr_gt_traj_x = [gt[f][k][0] for f in allFrames if f in gt.keys()]
                curr_gt_traj_y = [gt[f][k][1] for f in allFrames if f in gt.keys()]
                curr_err_traj_rec = [distances_rec[f][k] for f in allFrames if f in distances_rec.keys()]
                curr_err_traj_vol = [distances_vol[f][k] for f in allFrames if f in distances_vol.keys()]
                curr_err_traj_bb = [distances_bb[f][k] for f in allFrames if f in distances_bb.keys()]
                gt_traj_x.append( curr_gt_traj_x )
                gt_traj_y.append( curr_gt_traj_y )
                err_traj_rec.append( curr_err_traj_rec )
                err_traj_vol.append( curr_err_traj_rec )
                err_traj_bb.append( curr_err_traj_rec )
            except:
                break

            fig = plt.figure()
            plt.title(conf_title)
            plt.plot( list(distances_rec.keys()), curr_err_traj_rec )
            plt.plot( list(distances_vol.keys()), curr_err_traj_vol )
            plt.plot( list(distances_bb.keys()), curr_err_traj_bb )
            plt.xlabel('Frame')
            plt.ylabel('Localizing Accuracy [cm]')
            # plt.show()
            plt.savefig( join(outFolder, conf_title + ('_dists_over_frames_%01d.png' % k)  )  )

            curr_err_traj_rec_corr = [distances_rec[f][k] for f in allFrames if f in distances_rec.keys() & distances_vol.keys()]
            curr_err_traj_vol_corr = [distances_vol[f][k] for f in allFrames if f in distances_rec.keys() & distances_vol.keys()]
            fig = plt.figure()
            plt.title(conf_title)
            plt.scatter( curr_err_traj_vol_corr, curr_err_traj_rec_corr )            
            plt.xlabel('Localizing Accuracy [cm]')
            plt.ylabel('Localizing Accuracy [cm]')
            # plt.show()
            plt.savefig( join(outFolder, conf_title + ('_dists_corr_rec_vol_%01d.png' % k)  )  )
            
                
        # fig = plt.figure()
                
        # plt.scatter( gt_traj_x, gt_traj_y, c=err_traj_rec, cmap=matplotlib.cm.get_cmap('jet') )

        # plt.show()


        # plt.savefig( join(outFolder, conf_title + '_dist_rec.png')  )            
        



    with open(join(outFolder, 'compare2.csv'), 'w') as outF:
        outF.write(s)


