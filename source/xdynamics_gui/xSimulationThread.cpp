#include "xSimulationThread.h"
#include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include <QTime>
#include <QDebug>

xSimulationThread::xSimulationThread()
	: isStop(false)
	, xds(NULL)
{

}

xSimulationThread::~xSimulationThread()
{
	if (xds) delete xds; xds = NULL;
// 	if (dem) delete dem; dem = NULL;
// 	if (mbd) delete mbd; mbd = NULL;
}

bool xSimulationThread::xInitialize(xDynamicsManager* dm)
{
	if (!xds)
		xds = new xDynamicsSimulator(dm);
	if (!xds->xInitialize())
	{
		return false;
	}
	return true;
}

void xSimulationThread::setStopCondition()
{
	m_mutex.lock();
	isStop = true;
	m_mutex.unlock();
}

// bool xSimulationThread::savePart(double ct, unsigned int pt)
// {
// 	if (dem)
// 	{
// 		model::rs->insertTimeData(ct);
// 		double *v_pos = model::rs->getPartPosition(pt);
// 		//double *v_vel = model::rs->getPartVelocity(pt);
// 		QString part_name = dem->saveResult(v_pos, NULL, ct, pt);
// 		model::rs->insertPartName(part_name);
// 		//model::rs->definePartDatasDEM(false, pt);
// 	}
// 	if (mbd)
// 	{
// 		mbd->saveResult(ct);
// 	}
// 	return true;
// }

// bool xSimulationThread::saveFinalResult(double ct)
// {
// 	QString file = model::path + "/" + model::name + "_final.bfr";
// 	QFile qf(file);
// 	qf.open(QIODevice::WriteOnly);
// 	qf.write((char*)&ct, sizeof(double));
// 	if (dem)
// 	{
// 		dem->saveFinalResult(qf);
// 	}
// 	if (mbd)
// 	{
// 		mbd->saveFinalResult(qf);
// 	}
// 	qf.close();
// 	//dem->saveFinalResult(file);
// 	return true;
// }

void xSimulationThread::run()
{
	char buf[255] = { 0, };
	unsigned int part = 0;
	unsigned int cstep = 0;
	unsigned int eachStep = 0;
	unsigned int numPart = 0;
	double dt = xSimulation::dt;
	double ct = dt * cstep;
	// 	bool isUpdated = false;
	xSimulation::setCurrentTime(ct);
	double total_time = 0.0;
	double elapsed_time = 0.0;
	double previous_time = 0.0;
	QString ch;
	QTextStream qts(&ch);
	qts << "=========  =======    ==========    ======   ========   =============  ====================" << endl
		<< "PART       SimTime    TotalSteps    Steps    Time/Sec   TotalTime/Sec  Finish time         " << endl
		<< "=========  =======    ==========    ======   ========   =============  ====================" << endl;
	//xTime tme;
	QTime tme;
	QTime startTime = tme.currentTime();
	QDate startDate = QDate::currentDate();
	tme.start();
	sendProgress(part, ch);
	ch.clear();
	while (cstep < nstep)
	{
		QMutexLocker locker(&m_mutex);
		if (isStop)
			break;
		cstep++;
		cout << cstep << endl;
		eachStep++;
		ct += xSimulation::dt;
		xSimulation::setCurrentTime(ct);
		if (!xds->xRunSimulationThread(ct, cstep))
		{

		}

		if (!((cstep) % xSimulation::st))
		{
			previous_time = elapsed_time;
			elapsed_time = tme.elapsed() * 0.001;
			total_time += elapsed_time - previous_time;
			part++;
			if (xds->savePartData(ct, part))
			{
				ch.sprintf(buf, "Part%04d   %4.5f %10d      %5d      %4.5f     %4.5f    %ws", part, ctime, cstep, eachStep, elapsed_time - previous_time, total_time, xUtilityFunctions::GetDateTimeFormat("%d-%m-%y %H:%M:%S", 0).c_str());
				sendProgress(part, ch);
			}
			eachStep = 0;
		}
	}
	elapsed_time = tme.elapsed() * 0.001;
	total_time += elapsed_time;
	sendProgress(0, QString("=========  =======    ==========    ======   ========   =============  ====================\n"));
	xds->exportPartData();
	QTime endTime = QTime::currentTime();
	QDate endDate = QDate::currentDate();
	int minute = static_cast<int>(total_time / 60.0);
	int hour = static_cast<int>(minute / 60.0);
	double second = total_time - (hour * 3600 + minute * 60);
	qts << "     Starting time/date     = " + startTime.toString() + " / " + startDate.toString() << endl
		<< "     Ending time/date       = " + endTime.toString() + " / " + endDate.toString() << endl
		<< "     CPU + GPU time         = " + QString("%1").arg(total_time) + " second  ( " + QString("%1").arg(hour) + " h. " + QString("%1").arg(minute) + " m. " + QString("%1").arg(second) + " s. )" << endl;
	emit finishedThread();
}

