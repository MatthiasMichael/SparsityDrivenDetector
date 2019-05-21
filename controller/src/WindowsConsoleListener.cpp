#include "WindowsConsoleListener.h"


Q_DECLARE_METATYPE(HANDLE);

WindowsConsoleListener::WindowsConsoleListener(QObject * pParent):
	QObject(pParent),
	m_consoleNotifier(GetStdHandle(STD_INPUT_HANDLE))
{
	qRegisterMetaType<HANDLE>("HANDLE");
	connect(&m_consoleNotifier, &QWinEventNotifier::activated, this, &WindowsConsoleListener::processConsole);
}


void WindowsConsoleListener::processConsole(HANDLE h)
{
	INPUT_RECORD record;
	DWORD numRead;

	if (!ReadConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &record, 1, &numRead))
	{
		return;
	}

	if (record.EventType != KEY_EVENT || !record.Event.KeyEvent.bKeyDown)
	{
		return;
	}

	const char command = record.Event.KeyEvent.uChar.AsciiChar;

	emit consoleCommand(command);
}
