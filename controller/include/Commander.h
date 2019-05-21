#pragma once

#include <optional>

#include <QObject>

#include "Controller.h"
#include "WindowsConsoleListener.h"


class Commander : public QObject
{
	Q_OBJECT

public:
	using LocalCommand = void(*)(Commander * c);
	using MultiCommand = std::map<char, std::tuple<std::string, Controller::Command>>;

	Commander();

public slots:
	void processCommand(char c);
	void debugQuit();

private:
	void showHelp(std::ostream & os) const;

signals:
	void sendCommand(Controller::Command);

private:
	WindowsConsoleListener m_consoleListener;

	std::optional<MultiCommand> m_activeMultiCommand;

	std::map<char, std::tuple<std::string, Controller::Command>> m_controllerCommands;
	std::map<char, std::tuple<std::string, MultiCommand>> m_controllerMultiCommands;
	std::map<char, std::tuple<std::string, LocalCommand>> m_localCommands;
};