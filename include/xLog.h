#ifndef XLOG_H
#define XLOG_H

#include "xdynamics_decl.h"
//#include <QtCore/QFile>
//#include <QtCore/QTextStream>

class XDYNAMICS_API xLog
{
public:
	xLog();
	~xLog();

	static void releaseLogSystem();
	static void launchLogSystem(std::string d);
	static void log(std::string txt);
	static const char* getLogText();
private:
	static std::ofstream *qf;
	static char logtext[255];
};

#endif