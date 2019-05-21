# -*- coding: utf-8 -*-

import os

from config import Config
from config import Template

scene = 'C:/TEMP/SDD_Experiments/scenes/SimpleForPrecision.scene'

gridSizes = [(40, 40), (20, 20)]
gridHeight = 2.5

imageSize = (160, 120)

templates = [Template('res/person_01.ppm', 1, [60, 180], [120, 200])]
   
firstFrame = 4

detectorType = 'single'
maxMergingDistance = 60

advancedInits = [0, 1]
rootAlgs = ['Primal', 'Dual', 'Network', 'Barrier', 'Concurrent']

computeLocation = 'Device'

voxelSize = 30
minPartSegmentation = 0.5

maxClusterDistance = 100

c = 1

outDir = 'C:/TEMP/SDD_Experiments/config'


for gridSize in gridSizes:
    for advancedInit in advancedInits:
         for rootAlg in rootAlgs:
            config = Config(
                scene, 
                gridSize, 
                gridHeight, 
                imageSize, 
                templates, 
                firstFrame, 
                detectorType,
                maxMergingDistance,
                advancedInit, 
                rootAlg,
                computeLocation,
                voxelSize,
                minPartSegmentation,
                maxClusterDistance
                );
            fname = os.path.join(outDir, "{:04d}".format(c) + ".ini");
            c += 1
            with open(fname, 'w') as f:
                f.write(str(config))

