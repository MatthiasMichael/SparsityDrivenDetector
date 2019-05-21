#include "Context.h"

#include <opencv2/imgcodecs.hpp>

#include <boost/qvm/all.hpp>

#include "OptimizationProblemFactoryCollection.h"
#include "UnrealCoordinateSystem.h"
#include "environmentFromSceneInfo.h"
#include "Controller.h"
#include "VideoInput_Mpeg.h"
#include "OptimizationProblemTimer.h"
#include "conversion_helper.h"

#include "qvm_eigen.h"
#include "qvm_osg.h"
#include "qvm_cuda.h"


inline void preprocessing(const cv::Mat & input, cv::Mat & output)
{
	cv::Mat matResizedThresh;
	cv::threshold(input, matResizedThresh, 20, 1, cv::THRESH_BINARY);

	cv::Mat matResizedDilated;
	cv::dilate(matResizedThresh, matResizedDilated, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));

	matResizedThresh.convertTo(output, CV_32F);
};


std::vector<Template> makeTemplates
(
	const std::vector<std::string> & files, 
	const std::vector<int> & classes,
	const std::vector<Template::Size> & targetSizes,
	const std::vector<Template::Size> & maxSizes
)
{
	assert(files.size() == classes.size());
	assert(files.size() == targetSizes.size());
	assert(files.size() == maxSizes.size());

	std::vector<Template> templates;

	for (int i = 0; i < files.size(); ++i)
	{
		Template::Info info{ classes[i], targetSizes[i], maxSizes[i] };
		templates.emplace_back(cv::imread(files[i], cv::IMREAD_GRAYSCALE), info);
	}

	return templates;
}


Context::Elements Context::buildContext(const Configuration & config)
{
	UnrealCoordinateSystem::setDirections({ Vector3(1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, 1) });
	WorldCoordinateSystem::setDirections({ Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) });

	auto sceneInfo = SceneInfo::importScene(config.getSceneFile());

	Scene scene(sceneInfo);
	Environment environment(environmentFromSceneInfo(sceneInfo));

	const std::vector<Template> templates = makeTemplates(
		config.getTemplateFiles(), 
		config.getTemplateClasses(),
		config.getTemplateTargetSizes(),
		config.getTemplateMaxSizes()
	);

	const Dictionary dictionary(environment, templates, config.getGridParameters(), config.getTargetImageSize(), config.getSkipSDD());

	dictionary.saveDebugImages("C:/TEMP/");

	SparsityDrivenDetector detector(dictionary, makeFactory_withDecorator<OptimizationProblemTimer>(config.getOptimizationType()));
	
	auto & optimizationProblem = detector.tryGetOptimizationProblem();
	optimizationProblem.setCplexParam(IloCplex::Param::Advance, config.getAdvancedInitialization());
	optimizationProblem.setCplexParam(IloCplex::RootAlg, config.getRootAlgorithm());

	SparsityDrivenDetectorPostProcessing postProcessing({ config.getMaxMergingDistance() });

	std::unique_ptr<sfs::ShapeFromSilhouette_Impl> sfs_impl = sfs::makeShapeFromSilhouette(config.getComputeLocation());

	if (!config.getSkipSFS())
	{
		sfs_impl->setParameters(config.getSfsParameters());

		sfs_impl->createSpace
		(
			environment.getStaticMesh().getBoundingBox(),
			boost::qvm::convert_to<float3>(config.getVoxelSize()),
			environment.getCameras(),
			environment.getStaticMesh()
		);
	}
	
	auto input = std::make_unique<VideoInput_Mpeg>(sceneInfo, config.getTargetImageSize(), config.getTargetImageSize(),
	                                               &preprocessing, config.getFirstFrame());

	return 
	{ 
		std::move(sceneInfo), 
		std::move(scene), 
		std::move(environment), 
		std::move(detector),
		std::move(postProcessing),
		std::move(sfs_impl),
		std::move(input) 
	};
}


Context::Context(const Configuration & config) : Context(buildContext(config))
{
	// empty
}


Context::Context(Elements && elements) :
	m_sceneInfo(std::move(elements.m_sceneInfo)),
	m_scene(std::move(elements.m_scene)),
	m_environment(std::move(elements.m_environment)),
	m_detector(std::move(elements.m_detector)),
	m_postProcessing(std::move(elements.m_postProcessing)),
	m_sfs(std::move(elements.m_sfs)),
	m_input(std::move(elements.m_input))
{
	// empty
}
