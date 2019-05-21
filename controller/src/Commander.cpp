#include "Commander.h"

#include "ApplicationTimer.h"

#include "OptimizationProblemFactoryCollection.h"


Commander::Commander() :
	m_consoleListener(this), // Console listener must have the same thread affinity!
	m_controllerCommands
	{
		{
			'+', {
				"Next Frame", [](Controller * c)
				{
					c->nextFrame();
					c->processFrame();
				}
			}
		},
		{
			'-', {
				"Previous Frame", [](Controller * c)
				{
					c->previousFrame();
					c->processFrame();
				}
			}
		},
		{ 'o', { "Process Current Frame", [](Controller * c) { c->processFrame(); } } },
		{ 'r', { "Run", [](Controller * c) { c->processSequence(); } } },
		{ 'p', { "Pause", [](Controller * c) {} } },
		{ 'i', { "Info", [](Controller * c) { c->showInfo(std::cout); } } }
	},
	m_controllerMultiCommands
	{
		{
			'n', { "Switch Initialization Method",
				{
					{ '0', { "Value 0", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::Param::Advance, 0);
						}
					}},
					{ '1', { "Value 1", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::Param::Advance, 1);
						}}
					},
					{ '2', { "Value 2", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::Param::Advance, 2);
						}}
					}
				}
			}
		},
		{
			'a', { "Switch Root Algorithm",
				{
					{ '1', { "Primal", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::RootAlg, IloCplex::Algorithm::Primal);
						}
					}},
					{ '1', { "Dual", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::RootAlg, IloCplex::Algorithm::Dual);
						}}
					},
					{ '2', { "Barrier", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::RootAlg, IloCplex::Algorithm::Barrier);
						}}
					},
					{ '3', { "Concurrent", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::RootAlg, IloCplex::Algorithm::Concurrent);
						}}
					},
					{ '4', { "Network", [](Controller * c)
						{
							c->getDetector().tryGetOptimizationProblem().setCplexParam(IloCplex::RootAlg, IloCplex::Algorithm::Network);
						}}
					}
				}
			}
		},
		{
			's', { "Switch Optimization Problem",
				{
					{ '1', {"Single", [](Controller * c)
						{
							c->resetOptimizationProblem(std::make_unique<OptimizationProblemFactory_Single>());
							c->processFrame();
						}}
					},
					{ '2', {"SingleLayered", [](Controller * c)
						{
							c->resetOptimizationProblem(std::make_unique<OptimizationProblemFactory_SingleLayered>());
							c->processFrame();
						}}
					},
					{ '3', {"Multi", [](Controller * c)
						{
							c->resetOptimizationProblem(std::make_unique<OptimizationProblemFactory_Multi>());
							c->processFrame();
						}}
					},
					{ '4', {"MultiLayered", [](Controller * c)
						{
							c->resetOptimizationProblem(std::make_unique<OptimizationProblemFactory_MultiLayered>());
							c->processFrame();
						}}
					},
				}
			}
		}
	},
	m_localCommands
	{
		{ 'h', { "Help", [](Commander * c) { c->showHelp(std::cout); } } },
		{ 't', { "Show Timings", [](Commander * c) { AT_PRINT_NO_UNFINISHED(); } } }
	}
{
	connect(&m_consoleListener, &WindowsConsoleListener::consoleCommand, this, &Commander::processCommand);
	std::cout << "Awaiting command!" << std::endl;
}


void Commander::processCommand(char c)
{
	if(m_activeMultiCommand)
	{
		if (c == 'x')
		{
			std::cout << "Multi command canceled." << std::endl;
			m_activeMultiCommand = {};
			return;
		}

		try
		{
			auto command = m_activeMultiCommand.value().at(c);
			std::cout << "Issuing multi: " << std::get<0>(command) << " (" << c << ")" << std::endl;
			emit sendCommand(std::get<1>(command));
			m_activeMultiCommand = {};
		}
		catch (std::out_of_range &)
		{
			std::cout << std::string("Invalid option '") + c + "'. (Use 'x' to cancel.)" << std::endl;
		}

		return;
	}

	try
	{
		auto command = m_localCommands.at(c);
		std::cout << "Issuing local: " << std::get<0>(command) << " (" << c << ")" << std::endl;
		return std::get<1>(command)(this);
	}
	catch (std::out_of_range &)
	{
		// Do nothing. The command may still be in controller commands.
	}

	try
	{
		auto command = m_controllerCommands.at(c);
		std::cout << "Issuing: " << std::get<0>(command) << " (" << c << ")" << std::endl;
		emit sendCommand(std::get<1>(command));
		return;
	}
	catch (std::out_of_range &)
	{
		// Do nothing. The command may still be in controller multi commands.
	}

	try
	{
		auto command = m_controllerMultiCommands.at(c);
		std::cout << "Start multi command: " << std::get<0>(command) << " (" << c << ") -> ";
		m_activeMultiCommand = std::get<1>(command);
		for (auto [key, value] : m_activeMultiCommand.value())
		{
			std::cout << key << " ";
		}
		std::cout << std::endl;
		return;
	}
	catch (std::out_of_range &)
	{
		std::cout << "Invalid Command: " << c << std::endl;
	}
}


void Commander::debugQuit()
{
	std::cout << "\n\n\nCOMMANDER QUIT\n\n\n" << std::endl;
}


void Commander::showHelp(std::ostream & os) const
{
	os << "\tAvailable Commands:\n";

	for (const auto & [c, command] : m_localCommands)
	{
		os << "\t\t" << c << " -> " << std::get<0>(command) << std::endl;
	}

	for (const auto &[c, command] : m_controllerCommands)
	{
		os << "\t\t" << c << " -> " << std::get<0>(command) << std::endl;
	}

	for (const auto & [c, command] : m_controllerMultiCommands)
	{
		os << "\t\t" << c << " -> " << std::get<0>(command) << std::endl;
		for (const auto & [c_inner, command_inner] : std::get<1>(command))
		{
			os << "\t\t\t" << c_inner << " -> " << std::get<0>(command_inner) << std::endl;
		}
	}
}
