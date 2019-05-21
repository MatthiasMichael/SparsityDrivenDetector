#include "Controller.h"

#include <typeinfo>

#include <QApplication>
#include <QStyleFactory>

#include <QThread>
#include <QAbstractEventDispatcher>

#include "ApplicationTimer.h"

#include "OptimizationProblemDecorator.h"


Q_DECLARE_METATYPE(std::vector<cv::Mat>);
Q_DECLARE_METATYPE(Frame);
Q_DECLARE_METATYPE(Framenumber);
Q_DECLARE_METATYPE(Solution);
Q_DECLARE_METATYPE(MergedSolution);
Q_DECLARE_METATYPE(FusedSolution);
Q_DECLARE_METATYPE(Controller::Command);

Q_DECLARE_METATYPE(std::vector<const sfs::Voxel *>);
Q_DECLARE_METATYPE(std::vector<sfs::VoxelCluster>);

struct Controller::StringConstants
{
	inline static const auto TP_PROCESS = "Controller::ProcessInput";

	inline static const auto TP_PROCESS_INPUT = "Controller::ProcessInput::GetInput";
	inline static const auto TP_PROCESS_SDD = "Controller::ProcessInput::SparsityDrivenDetector";
	inline static const auto TP_PROCESS_SFS = "Controller::ProcessInput::ShapeFromSilhouette";
	inline static const auto TP_PROCESS_FUSION = "Controller::ProcessInput::Fusion";
};


Controller::Controller(Context && context) :
	m_sceneInfo(std::move(context.m_sceneInfo)),
	m_scene(std::move(context.m_scene)),
	m_environment(std::move(context.m_environment)),
	m_detector(std::move(context.m_detector)),
	m_postProcessing(std::move(context.m_postProcessing)),
	m_sfs(std::move(context.m_sfs)),
	m_input(std::move(context.m_input))
{
	qRegisterMetaType<std::vector<cv::Mat>>();
	qRegisterMetaType<Frame>();
	qRegisterMetaType<Framenumber>();
	qRegisterMetaType<Solution>();
	qRegisterMetaType<MergedSolution>();
	qRegisterMetaType<FusedSolution>();
	qRegisterMetaType<Command>();

	qRegisterMetaType<std::vector<const sfs::Voxel *>>();
	qRegisterMetaType<std::vector<sfs::VoxelCluster>>();
}


void Controller::resetInput()
{
	m_input->jumpToStart();
}


void Controller::setCplexParameters(CPXINT advancedInitialization, IloCplex::Algorithm rootAlg)
{
	auto & o = m_detector.tryGetOptimizationProblem();

	o.setCplexParam(IloCplex::RootAlg, rootAlg);
	o.setCplexParam(IloCplex::Param::Advance, advancedInitialization);
}


void Controller::showInfo(std::ostream & os) const
{
	os << "\tCurrent Framenumber: " << m_input->getCurrentFramenumber().get() << std::endl;
	
	try
	{
		const auto & decorator = dynamic_cast<const OptimizationProblemDecorator &>(m_detector.getIOptimizationProblem());
		os << "\tAccessing Problem via Decorator: " << typeid(m_detector.getIOptimizationProblem()).name() << std::endl;
	}
	catch(const std::bad_cast &)
	{
		// Do nothing
	}

	const auto & o = m_detector.tryGetOptimizationProblem();
	os << "\tAdvanced Initialization: " << o.getCplexParam(IloCplex::Param::Advance) << std::endl;
	os << "\tRoot Algorithm: " << o.getCplexParam(IloCplex::RootAlg) << std::endl;
	
	os << std::endl;
}


void Controller::executeCommand(Command c)
{
	c(this);
}


void Controller::processSequence()
{
	while (!QThread::currentThread()->eventDispatcher()->hasPendingEvents())
	{
		nextFrame();
		processFrame();
	}
}


void Controller::processFrame()
{
	AT_START(StringConstants::TP_PROCESS);
	emit beginProcessFrame();

	// Get Input
	AT_START(StringConstants::TP_PROCESS_INPUT);
	const Framenumber currentFramenumber = m_input->getCurrentFramenumber();
	const Frame frame = m_scene.getFrame(currentFramenumber);
	AT_STOP(StringConstants::TP_PROCESS_INPUT);

	// Call SDD
	AT_START(StringConstants::TP_PROCESS_SDD);
	const auto actors = m_detector.processFrame(m_input->getPreprocessedImages());
	const Solution solution{ frame.framenumber, frame.timestamp, actors };

	const auto mergedSolution = m_postProcessing.postProcessSolution(solution);
	AT_STOP(StringConstants::TP_PROCESS_SDD);

	// Call SfS
	AT_START(StringConstants::TP_PROCESS_SFS);
	if (m_sfs->hasSpace())
	{
		m_sfs->processInput(m_input->getImages());
	}
	AT_STOP(StringConstants::TP_PROCESS_SFS);

	// Fuse Results
	AT_START(StringConstants::TP_PROCESS_FUSION);
	const auto fusedSolution = fuse(mergedSolution, m_sfs->getCluster());
	AT_STOP(StringConstants::TP_PROCESS_FUSION);

	AT_STOP(StringConstants::TP_PROCESS);

	// Tell others what I got
	if (currentFramenumber < m_scene.getMaxFramenumber())
	{
		emit newSegmentationImages(m_input->getImagesScaledForDisplay());
		emit newGroundTruth(frame);
	}

	emit newSfsObjects(m_sfs->getActiveVoxels(), m_sfs->getCluster());

	if (actors.empty())
	{
		emit newEmptySolution();
		emit newEmptyMergedSolution();
		emit newEmptyFusedSolution();

		emit newEmptyReconstructedFrames();
	}
	else
	{
		emit newSolution(solution);
		emit newMergedSolution(mergedSolution);
		emit newFusedSolution(fusedSolution);
		
		emit newReconstructedImages(m_detector.getReconstructedFrames());
	}

	emit endProcessFrame();
}


void Controller::pause()
{
	// empty
}


void Controller::nextFrame()
{
	m_input->advanceFrame(true);
}


void Controller::previousFrame()
{
	m_input->jumpToFrame(Framenumber(m_input->getCurrentFramenumber().get() - 1));
}


void Controller::jumpToFrame(Framenumber n)
{
	m_input->jumpToFrame(n);
}


bool Controller::hasFrame()
{
	return m_input->getCurrentFramenumber().get() < m_input->getMaximumFramenumber().get() - 1;
}


void Controller::resetOptimizationProblem(const OptimizationProblem::Parameters & parameters)
{
	m_detector.resetOptimizationProblem(parameters);
}


void Controller::resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory,
                                          const OptimizationProblem::Parameters & parameters)
{
	m_detector.resetOptimizationProblem(std::move(factory), parameters);
}


void Controller::resetOptimizationProblem(std::unique_ptr<OptimizationProblemFactory> && factory)
{
	m_detector.resetOptimizationProblem(std::move(factory));
}


void Controller::reset(Context && context)
{
	// Input needs to be reset first so that it no longer references the video file in
	// the temporary folder of sceneInfo
	m_input = std::move(context.m_input);

	m_sceneInfo = std::move(context.m_sceneInfo);
	m_scene = std::move(context.m_scene);
	m_environment = std::move(context.m_environment);
	m_detector = std::move(context.m_detector);
}


void Controller::findOptimalOptimizationStrategy(int optimizationLength)
{
	std::vector<int> startMethods = { 0, 1, 2 };
	std::vector<IloCplex::Algorithm> rootAlgs =
	{
		IloCplex::Primal,
		IloCplex::Dual,
		IloCplex::Network,
		IloCplex::Barrier,
		IloCplex::Concurrent
		// Sifting seems to hang the application
	};

	for (int i = 0; i < rootAlgs.size(); ++i)
	{
		for (int j = 0; j < startMethods.size(); ++j)
		{
			std::stringstream ss;
			ss << "Config_" << i << "_" << j;
			const std::string timingName = ss.str();

			resetOptimizationProblem({ 0.1f });
			setCplexParameters(startMethods[j], rootAlgs[i]);
			resetInput();

			for (int f = 0; f < 20; ++f)
			{
				AT_START(timingName);
				processFrame();
				std::cout << timingName << " (" << f << "): " << AT_STOP_GET_TIME(timingName) << std::endl;
			}
		}
	}

	AT_PRINT();
}
