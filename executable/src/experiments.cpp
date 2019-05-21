#include <QApplication>
#include <QThread>

#include "boost/program_options.hpp"

#include "MainWindow.h"
#include "ExperimentCommander.h"
#include "application_helper.h"


struct options
{
	std::string experimentsFolder;
	std::string outputFolder;
	bool run;
};


options getOptions(int argc, char ** argv)
{
	using namespace boost::program_options;

	options_description commandLineOptions("Command Line Options");
	commandLineOptions.add_options()
		("help,h", "Show help")
		("experimentsFolder,e", value<std::string>(), "Folder containing the experiment configurations")
		("outputFolder,o", value<std::string>(), "Folder where the output files should be stored");

	variables_map vars_cmd;
	store(parse_command_line(argc, argv, commandLineOptions), vars_cmd);
	notify(vars_cmd);

	if (vars_cmd.count("help"))
	{
		std::cout << commandLineOptions << std::endl;
		return { "", "", false };
	}

	try
	{
		return
		{
			vars_cmd["experimentsFolder"].as<std::string>(),
			vars_cmd["outputFolder"].as<std::string>(),
			true
		};
	}
	catch(...)
	{
		std::cout << "Something went wrong fetching program options. Usage: " << std::endl;
		std::cout << commandLineOptions << std::endl;
	}
	return { "", "", false };
}


int main(int argc, char ** argv)
{
	const auto options = getOptions(argc, argv);

	if(!options.run)
	{
		return EXIT_SUCCESS;
	}

	QApplication app(argc, argv);
	MainWindow w;

	initVisuals("SDD_Experiments", &w);

	QThread workerThread;
	std::unique_ptr<Controller> c(nullptr);

	// The Controller has no default ctor so we need all info before we can construct it
	// It will also live in the same thread that constructed it so we have to connect
	// it to the main window in the main (GUI) thread. 
	const auto controllerFactory = [&c, &w, &workerThread](Context && ctx) -> Controller *
	{
		if (c)
		{
			c->reset(std::move(ctx));
		}
		else
		{
			c.reset(new Controller(std::move(ctx)));
			executeInMainThread([&]{ connectControllerToMainWindow(c.get(), &w); });
		}
		return c.get();
	};

	ExperimentCommander experimentCommander(options.experimentsFolder, options.outputFolder, controllerFactory);
	experimentCommander.moveToThread(&workerThread);

	QObject::connect(&workerThread, &QThread::started, &experimentCommander, &ExperimentCommander::execute);

	QObject::connect(&experimentCommander, &ExperimentCommander::finished, &app, &QApplication::quit);

	QObject::connect(&app, &QApplication::aboutToQuit, &workerThread, &QThread::quit);

	workerThread.start();
	
	const auto ret = app.exec();

	workerThread.wait();

	return ret;
}
