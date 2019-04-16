#ifndef XSIMULATIONTHREAD_H
#define XSIMULATIONTHREAD_H

// #include "dem_simulation.h"
// #include "multibodyDynamics.h"

#include <QThread>

class xDynamicsSimulator;
class xDynamicsManager;

class xSimulationThread : public QThread
{
	Q_OBJECT

public:
	xSimulationThread();
	~xSimulationThread();

	bool xInitialize(xDynamicsManager* _xdm);
	//bool xInitialize(QString xmlFile);

	unsigned int totalStep() { return nstep; }
	unsigned int totalPart() { return npart; }

	public slots:
	void setStopCondition();

private:
	void run() Q_DECL_OVERRIDE;

	//bool savePart(double ct, unsigned int pt);
	//bool saveFinalResult(double ct);

	bool isStop;
	unsigned int nstep;
	unsigned int npart;
	QMutex m_mutex;
	xDynamicsSimulator* xds;
// 	modelManager* mg;
// 	dem_simulation *dem;
// 	multibodyDynamics *mbd;

signals:
	void finishedThread();
	void sendProgress(int, QString, QString info = "");
	void excuteMessageBox();
};

#endif