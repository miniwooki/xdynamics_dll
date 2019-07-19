#include "xdynamics_gui.h"
#include <crtdbg.h>
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	if (AllocConsole())
	{
		freopen("CONIN$", "rb", stdin);
		freopen("CONOUT$", "wb", stdout);
		freopen("CONOUT$", "wb", stderr);
	}
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	QApplication a(argc, argv);
	xdynamics_gui w(argc, argv);
	w.show();
	return a.exec();
}
