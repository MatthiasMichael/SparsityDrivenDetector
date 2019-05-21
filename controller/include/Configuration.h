#pragma once

#include <string>
#include <vector>

#include "GridPoints.h"
#include "Template.h"
#include "TemplateTransformer.h"
#include "DisplayableFrame.h"
#include "ShapeFromSilhouetteBridge.h"

#ifndef IL_STD
#define IL_STD
#endif
#include <cstring>
#include <ilcplex/ilocplex.h>


class Configuration
{
public:
	Configuration(int argc, char ** argv);
	Configuration(const std::string configFilename);

	void initFromFilename(const std::string & configFilename);

	bool isConfigAvailable() const { return configAvailable; }

	const std::string & getSceneFile() const;

	const GridPoints::Parameters & getGridParameters() const;

	const TemplateTransformer::ImageSize & getTargetImageSize() const;

	const std::vector<std::string> & getTemplateFiles() const;
	const std::vector<int> & getTemplateClasses() const;
	const std::vector<Template::Size> & getTemplateTargetSizes() const;
	const std::vector<Template::Size> & getTemplateMaxSizes() const;

	const Framenumber & getFirstFrame() const;

	const std::string & getOptimizationType() const;
	ScalarType getMaxMergingDistance() const;

	CPXINT getAdvancedInitialization() const;
	IloCplex::Algorithm getRootAlgorithm() const;

	sfs::ComputeLocation getComputeLocation() const;
	Vector3 getVoxelSize() const;

	const sfs::ShapeFromSilhouette_Impl::Parameters & getSfsParameters() const;

	bool getSkipSDD() const;
	bool getSkipSFS() const;

	static Configuration tryMakeConfiguration(int argc, char ** argv);

private:
	Configuration();

	void ensureConfigAvailable() const;

private:
	bool configAvailable = false;

	std::string sceneFile;

	GridPoints::Parameters gridParameters = { make_named<GridPoints::Distance>(0, 0), 0 };

	TemplateTransformer::ImageSize targetImageSize;

	Framenumber firstFrame;

	std::string optimizationType;
	ScalarType maxMergingDistance;

	std::vector<std::string> templateFiles;
	std::vector<int> templateClasses;
	std::vector<Template::Size> templateTargetSizes;
	std::vector<Template::Size> templateMaxSizes;

	CPXINT advancedInitialization;
	IloCplex::Algorithm rootAlgorithm;

	sfs::ComputeLocation computeLocation;
	Vector3 voxelSize;

	sfs::ShapeFromSilhouette_Impl::Parameters sfsParameters;

	bool debug_skipSDDBuildStep;
	bool debug_skipSFSBuildStep;
};
