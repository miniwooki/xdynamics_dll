#include "xLog.h"
#include "xdynamics_manager/xModel.h"
//#include <QtCore/QDate>
//#include <QtCore/QTime>

std::ofstream *xLog::qf = NULL;

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
	/*QTime cTime = QTime::currentTime();
	QDate cDate = QDate::currentDate();
	QString l = ">> " + cTime.toString() + "/" + cDate.toString() + " - " + QString::fromStdString(txt);*/
	//QString k_l = kor(l.toLocal8Bit());
	//char* local8b = l.toLocal8Bit().data()
	//std::cout << l.toLocal8Bit().data() << std::endl;
	//(*qf) << l.toStdString().c_str() << std::endl;
//	QTextStream qts(&qf);
	//qts << l << endl;
}