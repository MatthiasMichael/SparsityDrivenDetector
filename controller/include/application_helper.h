#pragma once

#include <QApplication>
#include <QStyleFactory>

#include "Controller.h"
#include "MainWindow.h"
#include <QAbstractEventDispatcher>


inline void initVisuals(const char * name, MainWindow * pMainWindow)
{
	qApp->setApplicationName(name);
	qApp->setStyle(QStyleFactory::create("Fusion"));

	if (pMainWindow)
	{
		pMainWindow->setStyleSheet("background-color: #aaaaaa;");
		pMainWindow->setWindowTitle(qApp->applicationName());

		pMainWindow->showMaximized();
	}
}


inline void connectControllerToMainWindow(const Controller * c, const MainWindow * w)
{
	w->showStaticElements(c->getEnvironment());
	w->showGrid(c->getDetector().getGrid());

	QObject::connect(c, &Controller::newGroundTruth, w, &MainWindow::showGroundTruth);

	QObject::connect(c, &Controller::newSolution, w, &MainWindow::showSolution);
	QObject::connect(c, &Controller::newMergedSolution, w, &MainWindow::showMergedSolution);
	QObject::connect(c, &Controller::newFusedSolution, w, &MainWindow::showFusedSolution);

	QObject::connect(c, &Controller::newEmptySolution, w, &MainWindow::clearSolution);
	QObject::connect(c, &Controller::newEmptyMergedSolution, w, &MainWindow::clearMergedSolution);
	QObject::connect(c, &Controller::newEmptyFusedSolution, w, &MainWindow::clearFusedSolution);

	QObject::connect(c, &Controller::newSfsObjects, w, &MainWindow::showSfsObjects);

	using ImageFunction = void(MainWindow::*)(const std::vector<cv::Mat> &) const;

	QObject::connect(c, &Controller::newSegmentationImages, w,
	                 static_cast<ImageFunction>(&MainWindow::showSegmentationImages));
	QObject::connect(c, &Controller::newReconstructedImages, w,
	                 static_cast<ImageFunction>(&MainWindow::showReconstructedImages));

	QObject::connect(c, &Controller::newEmptyReconstructedFrames, w, &MainWindow::clearReconstructedImages);
}


inline void printUncaughtExceptionOnTerminate()
{
	const auto e = std::current_exception();

	try
	{
		std::rethrow_exception(e);
	}
	catch(const std::runtime_error & re)
	{
		std::cout << "Runtime Error occurred:" << std::endl;
		std::cout << re.what() << std::endl;
	}
	catch(const std::exception & ge)
	{
		std::cout << "General Exception occurred:" << std::endl;
		std::cout << ge.what() << std::endl;
	}
	catch(...)
	{
		std::cout << "Unknown Exception was thrown." << std::endl;
	}

	std::abort();
}


// Workaround for QMetaObject::invokeMethod, which is not available for lambdas until Qt5.10
template <typename F>
static void executeInMainThread(F && fun) 
{
	QObject src;
	QObject::connect(&src, &QObject::destroyed, qApp, std::forward<F>(fun), Qt::QueuedConnection);
}


template <typename F>
static void executeInThread(F && fun, QThread * thread) {
	QObject * obj = QAbstractEventDispatcher::instance(thread);
	Q_ASSERT(obj);
	QObject src;
	QObject::connect(&src, &QObject::destroyed, obj, std::forward<F>(fun), Qt::QueuedConnection);
}
