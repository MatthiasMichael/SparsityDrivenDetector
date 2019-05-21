#pragma once

#include <memory>

#include <QObject>

#include "Environment.h"
#include "Scene.h"
#include "SceneInfo.h"
#include "SparsityDrivenDetector.h"
#include "VideoInput.h"
#include "Context.h"

#include "ShapeFromSilhouetteBridge.h"
#include "SparsityDrivenDetectorPostProcessing.h"
#include "Fusion.h"


class Controller : public QObject
{
	Q_OBJECT

public:
	using Command = void(*)(Controller *);

	struct StringConstants;

	Controller(Context && context);

	void processSequence();
	void processFrame();
	void pause();
	void findOptimalOptimizationStrategy(int optimizationLength);

	void resetInput();
	void nextFrame();
	void previousFrame();
	void jumpToFrame(Framenumber n);
	bool hasFrame();

	void resetOptimizationProblem(const OptimizationProblem::Parameters & parameters);
	void resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory, const OptimizationProblem::Parameters & parameters);
	void resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory);

	void reset(Context && context);

	void setCplexParameters(CPXINT advancedInitialization, IloCplex::Algorithm rootAlg);

	void showInfo(std::ostream & os) const;

	Framenumber getCurrentFramenumber() const { return m_input->getCurrentFramenumber(); }
	Framenumber getMaximumFramenumber() const { return m_input->getMaximumFramenumber(); }

	const Environment & getEnvironment() const { return m_environment; }
	const SparsityDrivenDetector & getDetector() const { return m_detector; }

public slots:
	void executeCommand(Command c);

signals:
	void beginProcessFrame();
	void endProcessFrame();

	void newSegmentationImages(const std::vector<cv::Mat> & images);
	void newGroundTruth(const Frame & f);

	void newSolution(const Solution & s);
	void newMergedSolution(const MergedSolution & s);
	void newFusedSolution(const FusedSolution & s);

	void newReconstructedImages(const std::vector<cv::Mat> & images);

	void newEmptySolution();
	void newEmptyMergedSolution();
	void newEmptyFusedSolution();

	void newEmptyReconstructedFrames();

	void newSfsObjects(const std::vector<const sfs::Voxel *> &, const std::vector<sfs::VoxelCluster> &);
	
private:
	SceneInfo m_sceneInfo;
	
	Scene m_scene;
	Environment m_environment;
	SparsityDrivenDetector m_detector;
	SparsityDrivenDetectorPostProcessing m_postProcessing;

	std::unique_ptr<sfs::ShapeFromSilhouette_Impl> m_sfs;

	std::unique_ptr<VideoInput> m_input;
};
