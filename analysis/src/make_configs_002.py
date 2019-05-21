# -*- coding: utf-8 -*-

import os

from config import Config
from config import Template

scene = 'C:/TEMP/SDD_Experiments/scenes/SimpleForPrecision.scene'

gridSizes = [(40, 40), (30, 30), (20, 20), (10, 10)]
gridHeight = 2.5

imageSize = (160, 120)

templates = [Template('res/person_01.ppm', 1, [60, 180], [120, 200])]
   
firstFrame = 4

detectorType = 'single'
maxMergingDistances = [100, 40]

advancedInit = 1
rootAlgs = rootAlg = 'Dual'

computeLocation = 'Device'

voxelSizes = [30, 20]
minPartSegmentation = 0.05

maxClusterDistance = 100

c = 1

outDir = 'C:/TEMP/SDD_Experiments/config'


for gridSize in gridSizes:
    for maxMergingDistance in maxMergingDistances:
         for voxelSize in voxelSizes:
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

