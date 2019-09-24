#include "xLog.h"
#include "xdynamics_manager/xModel.h"
#include <chrono>
#include <ctime>
//#include <QtCore/QDate>
//#include <QtCore/QTime>

std::ofstream *xLog::qf = NULL;
char xLog::logtext[] = "";

xLog::xLog()
{
 	//log_path = QString::fromStdString(d);
//  	if (qf && qf->is_open())
// 	{
// 		if (qf.fileName() != qd)
// 		{
// 			qf.close();
// 			qf.setFileName(qd);
// 		}
// 	}
// 	else 

	/*qf.open(QIODevice::WriteOnly);*/
	//qts.setDevice(&qf);
}

xLog::~xLog()
{

}

void xLog::releaseLogSystem()
{
	if (qf)
	{
		qf->close();
		delete qf;
	}
}

void xLog::launchLogSystem(std::string d)
{
	qf = new std::ofstream;
	d += "log.txt";
	qf->open(d, ios::out);
}

void xLog::log(std::string txt)
{
	chrono::system_clock::time_point now_t = chrono::system_clock::now();
	std::time_t now_time = std::chrono::system_clock::to_time_t(now_t);
	sprintf_s(logtext, "Log text - %s", txt.c_str());
	(*qf) << std::ctime(&now_time) << " - " << txt << std::endl;
//	QTextStream qts(&qf);
	//qts << l << endl;
}

const char * xLog::getLogText()
{
	return logtext;
}
