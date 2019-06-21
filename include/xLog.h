#ifndef XLOG_H
#define XLOG_H

#include "xdynamics_decl.h"
#include <QtCore/QFile>
#include <QtCore/QTextStream>

class XDYNAMICS_API xLog
{
public:
	xLog();
	~xLog();

	static void releaseLogSystem();
	static void launchLogSystem(std::wstring d);
	static void log(std::wstring txt);

private:
	static std::ofstream *qf;
};

#endif