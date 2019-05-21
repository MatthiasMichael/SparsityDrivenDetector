# -*- coding: utf-8 -*-

import os
from os import listdir
from os.path import isfile, join
import shutil
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np




if __name__=='__main__':
    csv_file = r'C:\TEMP\SDD_Experiments\our_experiment_003_fixed\figures\results.csv'

    matplotlib.rc('font', **{'family': 'serif'})
    matplotlib.rcParams['text.usetex'] = True

    errors = dict()
    precisions = dict()
    recalls = dict()
    timings = dict()

    heights = set()
    optims = set()
    img_sizes = set()
    merge_dists = set()

    with open(csv_file, 'r') as f_in:
        f_in.readline() #spaltenueberschriften
        for line in f_in:
            line = line.split(';')

            height = float(line[0])
            optim = line[1]
            img_size = line[2]
            merge_dist = float(line[3])

            heights.add(height)
            optims.add(optim)
            img_sizes.add(img_size)
            merge_dists.add(merge_dist)

            errors[ (height, optim, img_size, merge_dist) ] = float(line[4])
            precisions[ (height, optim, img_size, merge_dist) ] = float(line[5])
            recalls[ (height, optim, img_size, merge_dist) ] = float(line[6])
            timings[ (height, optim, img_size, merge_dist) ] = float(line[7])




    #plt.figure()
    #plt.xlabel('Camera height above ground [cm]')
    #plt.ylabel('Localization error [cm]')

    #legend_labels = []
    #legend_handles = []

    #for optim in optims:
    #    for img_size in img_sizes:
    #        for merge_dist in merge_dists:
    #            plt_heights_x = []
    #            plt_errors_y = []
    #            for height in heights:
    #                plt_heights_x.append(height)
    #                plt_errors_y.append( errors[ (height, optim, img_size, merge_dist) ] )
    #            legend, = plt.plot( plt_heights_x, plt_errors_y )
    #            legend_handles.append(legend)
    #            legend_labels.append( '{}_{}_{}_{}'.format( height, optim, img_size, merge_dist ) )

    #plt.legend( legend_handles, legend_labels )

    #plt.ylim([0, 1.1 * max(errors.values())])

   
    #plt.show()






    fig, ax1 = plt.subplots()

    ax2 = plt.twinx()
    colors = ('#2e6284', '#8bbbda')

    legend_labels = []
    legend_handles = []

    styles = ['*', 'X', 's', 'd']

    for (optim, style) in zip(optims, styles):
        plt_heights_x = []
        plt_errors_y = []
        plt_timings_y = []
        for height in heights:
            plt_heights_x.append(height)
            plt_errors_y.append( errors[ (height, optim, '160x120', 100.) ] )
            plt_timings_y.append( timings[ (height, optim, '160x120', 100.) ] )
        legend, = ax1.plot( plt_heights_x, plt_errors_y, color=colors[0], marker=style )
        legend_handles.append(legend)
        # legend_labels.append( '{}_{}_{}_{}'.format( height, optim, img_size, merge_dist ) )
        legend_labels.append( optim )
        ax2.plot( plt_heights_x, plt_timings_y, color=colors[1], marker=style )

    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax1.set_xlabel('Camera height above ground (cm)')
    ax2.set_ylabel('Execution time (ms)', color=colors[0])
    ax1.set_ylabel('Localization error (cm)', color=colors[0])
    ax2.set_ylim( [0, 1.1 * max(timings.values())] )

    plt.legend( legend_handles, legend_labels )


    ax1.set_ylim([0, 1.1 * max(errors.values())])
    fig.set_size_inches(5,3.5);

    plt.savefig(r'C:\TEMP\SDD_Experiments\our_experiment_003_fixed\figures\cam_height.eps', transparent=True, bbox_inches='tight', dpi=300)


    plt.show()


    for optim in optims:

        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111, projection='3d')

        #fake data

        img_sizes_x = []
        merge_dists_y = []
        errors_z = []

        for (x, img_size) in enumerate(img_sizes):
            for (y, merge_dist) in enumerate(merge_dists):   
                img_sizes_x.append(x)
                merge_dists_y.append(y)
                errors_z.append( errors[ (200., optim, img_size, merge_dist) ] ) 

        top = errors_z
        bottom = np.zeros_like(top)
        width = depth = 1

        ax.set_zlabel('Localization error [cm]')
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(list(img_sizes))
        ax.set_yticklabels(list(merge_dists))
        ax.set_yticks([0.5, 1.5])
        ax.set_ylabel('Merge distance [cm]')

        ax.bar3d(img_sizes_x, merge_dists_y, bottom, width, depth, top)
        ax.set_title(optim)


        plt.show()    




    for optim in optims:

         fig = plt.figure(figsize=(8, 3))
         ax = fig.add_subplot(111, projection='3d')

         #fake data

         img_sizes_x = []
         merge_dists_y = []
         timings_z = []

         for (x, img_size) in enumerate(img_sizes):
             for (y, merge_dist) in enumerate(merge_dists):   
                 img_sizes_x.append(x)
                 merge_dists_y.append(y)
                 timings_z.append( timings[ (200., optim, img_size, merge_dist) ] ) 

         top = timings_z
         bottom = np.zeros_like(top)
         width = depth = 1

         ax.set_zlabel('Localization error [cm]')
         ax.set_xticks([0.5, 1.5])
         ax.set_xticklabels(list(img_sizes))
         ax.set_yticklabels(list(merge_dists))
         ax.set_yticks([0.5, 1.5])
         ax.set_ylabel('Merge distance [cm]')

         ax.bar3d(img_sizes_x, merge_dists_y, bottom, width, depth, top)
         ax.set_title(optim)


         plt.show()    





      ## # # # # Plotting
      #plt.style.use('seaborn-paper')
    
      #fig, ax1 = plt.subplots()
      ## plt.ylim([0, 600])
      #ax1.ylabel('Localization error [cm]')
      #ax2 = plt.twinx()
      #ax2.ylabel('Execution time [ms]')


      #rootAlg2realWord = dict( zip(allRootAlgs, allRootAlgs) )

      ##plt.title(title)
      ##plt.xscale(0.5)
      #width = 0.4  # the width of the bars
      #ind = np.arange( len(img_sizes) * len(merge_dists)  ) 

      #rects = dict()
      #colors = ('#2e6284', '#8bbbda')

      #for idx, (rootAlg, currColor) in enumerate(zip(sorted(allRootAlgs), colors)):
      #    means_rootAlg = ( np.mean( seqData[0][rootAlg].cplex_durations ), np.mean( seqData[1][rootAlg].cplex_durations ) )
      #    stds_rootAlg = ( np.std( seqData[0][rootAlg].cplex_durations ), np.std( seqData[1][rootAlg].cplex_durations ) )
   
      #    x = ind - width/2 + width/(len(allRootAlgs) - 1) * idx
      #    #rects[rootAlg] = ax.bar(x, means_rootAlg, width/len(allRootAlgs), yerr=stds_rootAlg, label=rootAlg, zorder=3, capsize=10, error_kw={'zorder':5})
      #    rects[rootAlg] = ax.bar(x, means_rootAlg, width/len(allRootAlgs), color=currColor, label=rootAlg, zorder=2)
      #    for j in range(len(x)):
      #        (_,caps,_) = ax.errorbar(x[j], means_rootAlg[j], stds_rootAlg[j], ecolor='black', capsize=3, linewidth=1, zorder=5)

      #        for cap in caps:
      #            cap.set_color('black')
      #            cap.set_markeredgewidth(1)

        

      #    #autolabel(ax, rects[rootAlg], "center")

      #ax.set_xticks(ind)
      #ax.set_xticklabels(('Zero Initialization', 'Advanced Initialization'))
      ##ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
      #legend = ax.legend(loc='upper left', fancybox=False, framealpha=True);
      #legend.get_frame().set_zorder(5);
      #legend.get_frame().set_linewidth(1);
      #legend.get_frame().set_facecolor('white')
      #legend.get_frame().set_edgecolor('black')


      #mng = plt.get_current_fig_manager()
      ##mng.full_screen_toggle()
    
      #fig.set_size_inches(5,3);
      ##ax.subplots_adjust(right = 0.5)
      #plt.savefig(join(outFolder, "timing_barplot.eps"), transparent=True, bbox_inches='tight', dpi=300)
   
      #plt.show()    