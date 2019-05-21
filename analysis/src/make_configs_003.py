# -*- coding: utf-8 -*-

import os

from config import Config
from config import Template

scene = 'C:/TEMP/SDD_Experiments/scenes/Occlusions.scene'

gridSize = (20, 20)
gridHeight = 2.5

imageSizes = [(80, 60), (160, 120)]

templates = [Template('res/person_01.ppm', 1, [60, 180], [120, 200]), Template('res/person_02.ppm', 1, [60, 180], [120, 200])]
   
firstFrame = 4

detectorTypes = ['single', 'singleLayered', 'multi', 'multiLayered']
maxMergingDistance = 100

advancedInit = 1
rootAlgs = rootAlg = 'Dual'

computeLocation = 'Device'

voxelSize = 30
minPartSegmentation = 0.05

maxClusterDistance = 3

c = 1

outDir = 'C:/TEMP/SDD_Experiments/config'


for imageSize in imageSizes:
        for detectorType in detectorTypes:
    
        
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

