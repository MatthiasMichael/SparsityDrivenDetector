#include <iostream>

#include <QApplication>
#include <QThread>

#include "ApplicationTimer.h"

#include "Controller.h"

#include "Commander.h"
#include "Configuration.h"
#include "Context.h"
#include "MainWindow.h"

#include "application_helper.h"


int main(int argc, char ** argv)
{
	const auto config = Configuration::tryMakeConfiguration(argc, argv);

	if (!config.isConfigAvailable())
	{
		std::cout << "Could not load config. Exiting." << std::endl;
		return EXIT_SUCCESS;
	}

	QApplication app(argc, argv);

	MainWindow w;

	initVisuals("SDD_Interactive", &w);

	auto c = Controller(Context(config));

	QThread t;
	c.moveToThread(&t);

	connectControllerToMainWindow(&c, &w);

	QThread t2;
	Commander commander;
	commander.moveToThread(&t2);

	QObject::connect(&commander, &Commander::sendCommand, &c, &Controller::executeCommand);

	QObject::connect(&app, &QApplication::aboutToQuit, &t, &QThread::quit);
	QObject::connect(&app, &QApplication::aboutToQuit, &t2, &QThread::quit);

	QObject::connect(&app, &QApplication::aboutToQuit, [&t] { executeInThread([] {}, &t); }); // Just to be sure?
	
	t.start();
	t2.start();
	
	const auto ret = app.exec();

	t.wait();
	t2.wait();

	AT_PRINT();

	return ret;
}
