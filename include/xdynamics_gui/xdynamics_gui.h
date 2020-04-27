#pragma once

#include <QtWidgets/QMainWindow>
#include <QtCore/QList>
#include <QProgressBar>
#include <QDockWidget>
#include <QToolBar>
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "ui_xdynamics_gui.h"
#include "xCommandWindow.h"
#include "xGLWidget.h"
#include "xAnimationTool.h"


class xDynamicsManager;
class xParticleManager;
class xModelNavigator;
class xCommandLine;
//class xGLWidget;
class wsimulation;
class wpointmass;
class wresult;
class xChartWindow;
class xColorControl;
class gen_cluster_dlg;
class wgenclusters;
class xObjectManager;
class xCheckCollisionDialog;
//class xSimulationThread;

class xdynamics_gui : public QMainWindow
{
	Q_OBJECT

public:
	enum { NEW = 0, OPEN, SAVE };
	enum { CUBE = 3, CYLINDER, CHART, UPLOAD_RESULT, 
		PASSING_DISTRIBUTION, CONVERT_MESH_TO_SPHERE, GENERATE_CLUSTER_DISTRIBUTION, 
		CHECK_COLLISION, REPOSITIONING_CLUSTERS, COLOR_FOR_PARTICLE_SIZE };
	
	xdynamics_gui(int _argc, char** _argv, QWidget *parent = Q_NULLPTR);
	~xdynamics_gui();
	static xdynamics_gui* XGUI();
	bool ReadViewModel(QString path);
	QString ReadXLSFile(QString path);
	void ReadSTLFile(QString path);
	bool ReadModelResults(QString path);
	void OpenFile(QString path);
	QSize FullWindowSize();
	//void xInitializeGUI(int _argc, char** _argv);
	void Clear();

private slots:
	void xNew();
	void xSave();
	void xOpen();
	void xCylinder();
	void xCube();
	void xChart();
	void xGetSimulationWidget(wsimulation*);
	void xGetPointMassWidget(wpointmass*);
	void xGetResultWidget(wresult*);
	void xGetGenerateClustersWidget(wgenclusters*);
	void xRunSimulationThread(double, unsigned int, double);
	void xExitSimulationThread();
	void xRecieveProgress(int, QString);
	void xEditCommandLine();
	void xGeometrySelection(QString);
	void xReleaseOperation();
	void xInitializeWidgetStatement();
	void xOnGeometrySelectionOfPointMass();
	void xStopSimulationThread();
	void xReleaseResultCallThread();
	void xContextMenuProcess(QString nm, contextMenuType vot);
	void xSetupResultNavigatorByChangeTargetCombo(int);
	void xUploadResultThisModel();
	void xSetupParticleBufferColorDistribution(int);
	void xSelectStartPoint();
	void xPassDistribution();// click_passing_distribution
	void xGenerateClusterDistribution();
	void xGenerateClusterParticles(QString, QString, int*, double*, bool);
	void xCheckCollision();
	void xExportClusterParticleModel();
	void xRepositioningClusters();
	void xColorForParticleSize();
	//void xHighlightParticle(unsigned int, QColor, QColor&);
	
private:
	void setupMeshSphere();
	void setupMainOperations();
	void setupObjectOperations();
	void setupAnimationTool();
	void setupBindingPointer();
	void setupMenuOperations();
	void setupShorcutOperations();
	//void setupAnimationOperations();
	void deleteFileByEXT(QString ext);
	void dragEnterEvent(QDragEnterEvent *event);
	void dropEvent(QDropEvent *event);

	bool isOnViewModel;
	xAnimationTool* myAnimationBar;
	QToolBar* myObjectBar;
	QList<QAction*> myMainActions;
	QList<QAction*> myObjectActions;
	
	//xSimulationThread* simThread;
	int caction;
	QProgressBar *pbar;
	xDynamicsManager* xdm;
	Ui::xdynamics_gui_mw ui;
	QString path;
	xGLWidget* xgl;
	xCommandWindow* xcw;
	xModelNavigator* xnavi;
	QDockWidget* xcomm;
	xCommandLine* xcl;
	xChartWindow* xchart;
	xColorControl* xcc;
	QSize desktop_size;
	QStringList result_file_list;
	xParticleManager* pmgr;
	xObjectManager* omgr;
	xCheckCollisionDialog* c_checker;
	//gen_cluster_dlg* gen_cluster;
};
