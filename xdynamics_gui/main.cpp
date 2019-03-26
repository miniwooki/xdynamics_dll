#include "xdynamics_gui.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	xdynamics_gui w(argc, argv);
	w.show();
	return a.exec();
}
