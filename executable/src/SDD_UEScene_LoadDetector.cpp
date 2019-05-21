#include <QApplication>
#include <QThread>

#include <opencv2/imgcodecs.hpp>

#include "enumerate.h"

#include "environmentFromSceneInfo.h"

#include "OptimizationProblemDecorator.h"

#include "UnrealCoordinateSystem.h"
#include "Controller.h"
#include "OptimizationProblemFactory_Single.h"
#include "VideoInput_ImageSequence.h"
#include "VideoInput_Mpeg.h"
#include "application_helper.h"

#include "MainWindow.h"
#include "OptimizationProblemTimer.h"


inline void preprocessing(const cv::Mat & input, cv::Mat & output)
{
	cv::Mat matResizedThresh;
	cv::threshold(input, matResizedThresh, 20, 1, cv::THRESH_BINARY);

	cv::Mat matResizedDilated;
	cv::dilate(matResizedThresh, matResizedDilated, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));

	matResizedThresh.convertTo(output, CV_32F);
};


int main(int argc, char ** argv)
{
	QApplication app(argc, argv);

	// BEGIN SETUP
	UnrealCoordinateSystem::setDirections({ Vector3(1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, 1) });
	WorldCoordinateSystem::setDirections({ Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) });

	// PARAMETERS
	const std::string sceneFile = "C:/TEMP/test002.scene";
	const std::string detectorFile = "C:/TEMP/Detector.zip";

	const auto startFrameNumber = make_named<Framenumber>(4);

	// BUILD DETECTOR
	auto sceneInfo = SceneInfo::importScene(sceneFile);
	auto scene = Scene(sceneInfo);

	auto environment = environmentFromSceneInfo(sceneInfo);

	auto detector = SparsityDrivenDetector::load(detectorFile);

	auto input = std::make_unique<VideoInput_Mpeg>(sceneInfo, detector.getDictionary().getTargetImageSize(), 
		detector.getDictionary().getTargetImageSize(), &preprocessing, startFrameNumber);

	// END SETUP

	MainWindow w;
	initVisuals("SDD_UEScene_LoadDetector", &w);

	QThread t;

	Controller c(Context({ std::move(sceneInfo), std::move(scene), std::move(environment), std::move(detector), std::move(input) }));
	c.moveToThread(&t);

	connectControllerToMainWindow(&c, &w);

	QObject::connect(&t, &QThread::started, [&c] { c.processSequence(); });

	QObject::connect(&app, &QApplication::aboutToQuit, &t, &QThread::quit);

	// Somehow just invoking the quit slot on t does not lead to an event being posted in the 
	// thread's event queue. To stop execution of c.processSequence() we post an empty event
	// to t so that c stops his current loop.
	QObject::connect(&app, &QApplication::aboutToQuit, [&t] { executeInThread([] {}, &t); });

	t.start();

	const auto ret = app.exec();

	t.wait();

	AT_PRINT();

	return ret;
}
