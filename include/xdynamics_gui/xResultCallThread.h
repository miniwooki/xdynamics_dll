#ifndef XRESULTCALLTHREAD_H
#define XRESULTCALLTHREAD_H

// #include "dem_simulation.h"
// #include "multibodyDynamics.h"

#include <QThread>
#include <QMutex>
#include <QStringList>

class xDynamicsManager;

class xResultCallThread : public QThread
{
	Q_OBJECT

public:
	xResultCallThread();
	~xResultCallThread();

	void set_dynamics_manager(xDynamicsManager* _xdm, QString fpath);
	QStringList get_file_list();

private:
	void run() Q_DECL_OVERRIDE;

	//bool savePart(double ct, unsigned int pt);
	//bool saveFinalResult(double ct);

	bool is_success_loading_model;
	QMutex m_mutex;
	xDynamicsManager* xdm;
	QString path;
	//xDynamicsSimulator* xds;
	//unsigned int last_pt;
	// 	modelManager* mg;
	// 	dem_simulation *dem;
	// 	multibodyDynamics *mbd;
	QStringList flist;

signals:
	void result_call_finish();
	void result_call_send_progress(int, QString);

	//	void excuteMessageBox();
};

#endif