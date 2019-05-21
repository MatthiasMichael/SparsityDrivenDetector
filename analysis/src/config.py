# -*- coding: utf-8 -*-

class Template:
    def __init__(self, file, classes, size, maxSize):
        self.file = file
        self.classes = classes
        self.size = size
        self.maxSize = maxSize

    def __str__(self):
        with open('template_template.txt', 'r') as f:
            s = f.read()
            return s.format(self.file, self.classes, self.size[0], self.size[1], self.maxSize[0], self.maxSize[1])

class Config:
    def __init__(self, scene, gridSize, gridHeight, imageSize, templates, firstFrame, 
                 optimizationType, maxMergingDistance, advancedInitialization, 
                 rootAlg, computeLocation, voxelSize, minPartSegmentation, maxClusterDistance):
        self.scene = scene
        self.gridSize = gridSize
        self.gridHeight = gridHeight
        self.imageSize = imageSize
        self.templates = templates
        self.firstFrame = firstFrame
        self.optimizationType = optimizationType
        self.maxMergingDistance = maxMergingDistance
        self.advancedInitialization = advancedInitialization
        self.rootAlg = rootAlg
        self.computeLocation = computeLocation
        self.voxelSize = voxelSize
        self.minPartSegmentation = minPartSegmentation
        self.maxClusterDistance = maxClusterDistance

    def __str__(self):
        with open('config_template.txt', 'r') as f:
            s = f.read()
            return s.format(self.scene, 
                     self.gridSize[0], self.gridSize[1], 
                     self.gridHeight,
                     self.imageSize[0], self.imageSize[1],
                     '\n\n'.join([str(t) for t in self.templates]),
                     self.firstFrame,
                     self.optimizationType,
                     self.maxMergingDistance,
                     self.advancedInitialization,
                     self.rootAlg,
                     self.computeLocation,
                     self.voxelSize,
                     self.minPartSegmentation,
                     self.maxClusterDistance)


