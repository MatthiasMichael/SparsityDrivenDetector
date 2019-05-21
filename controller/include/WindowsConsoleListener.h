#pragma once

#include <Windows.h>

#include <QObject>
#include <QWinEventNotifier>


class WindowsConsoleListener : public QObject
{
	Q_OBJECT

public:
	WindowsConsoleListener(QObject * pParent);

public slots:
	void processConsole(HANDLE h);

signals:
	void consoleCommand(char);

private:
	QWinEventNotifier m_consoleNotifier;
};