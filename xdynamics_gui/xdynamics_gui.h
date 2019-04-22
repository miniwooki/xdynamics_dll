#pragma once

#include <QtWidgets/QMainWindow>
#include <QtCore/QList>
#include <QProgressBar>
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "ui_xdynamics_gui.h"
#include "xCommandWindow.h"
//#include "xGLWidget.h"
#include "xAnimationTool.h"

class xDynamicsManager;
class xModelNavigator;
class xGLWidget;
class wsimulation;
//class xSimulationThread;

class xdynamics_gui : public QMainWindow
{
	Q_OBJECT

public:
	enum { NEW = 0, OPEN, SAVE };
	
	xdynamics_gui(int _argc, char** _argv, QWidget *parent = Q_NULLPTR);
	~xdynamics_gui();
	static xdynamics_gui* XGUI();
	bool ReadViewModel(QString path);
	QString ReadXLSFile(QString path);
	bool ReadModelResults(QString path);

private slots:
	void xNew();
	void xSave();
	void xOpen();
	void xGetSimulationWidget(wsimulation*);
	void xRunSimulationThread(double, unsigned int, double);
	void xExitSimulationThread();
	void xRecieveProgress(int, QString);
	
private:
	void setupMainOperations();
	void setupAnimationTool();
	//void setupAnimationOperations();
	
	void dragEnterEvent(QDragEnterEvent *event);
	void dropEvent(QDropEvent *event);

	bool isOnViewModel;
	xAnimationTool* myAnimationBar;
	QList<QAction*> myMainActions;
	
	//xSimulationThread* simThread;
	QProgressBar *pbar;
	xDynamicsManager* xdm;
	Ui::xdynamics_gui_mw ui;
	QString path;
	xGLWidget* xgl;
	xCommandWindow* xcw;
	xModelNavigator* xnavi;
};
