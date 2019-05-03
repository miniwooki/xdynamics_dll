#include "xdynamics_gui.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include "xModelNavigator.h"
#include "xGLWidget.h"
#include "xSimulationThread.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xNewDialog.h"
#include "xLineEditWidget.h"
#include "xCommandLine.h"
#include "xChartWindow.h"
//#include "xPointMassWidget.h"
#include <QtCore/QDir>
#include <QtCore/QMimeData>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QShortcut>

xdynamics_gui* xgui;
wsimulation* wsim;
wpointmass* wpm;
static xSimulationThread* sThread = NULL;

xdynamics_gui* xdynamics_gui::XGUI()
{
	return xgui;
}

xdynamics_gui::xdynamics_gui(int _argc, char** _argv, QWidget *parent)
	: QMainWindow(parent)
	, xgl(NULL)
	, xcw(NULL)
	, xnavi(NULL)
	, xdm(NULL)
	, pbar(NULL)
	, xcomm(NULL)
	, xcl(NULL)
	, xchart(NULL)
	, myAnimationBar(NULL)
	//, simThread(NULL)
	, isOnViewModel(false)
{
	xgui = this;
	path = QString::fromLocal8Bit(getenv("USERPROFILE"));
	path += "/Documents/xdynamics/";
	if (_argc > 1)
	{
		QString model_name = _argv[2];
		QString _path = path + model_name;
		if (!QDir(_path).exists())
		{
			qDebug() << "\"" << _path << "\"" << " is not exist.";
			exit(0);
		}	
		else
		{
			path = _path;
		}
	}
	
	ui.setupUi(this);
	xgl = new xGLWidget(_argc, _argv, NULL);
	xcw = new xCommandWindow(this);
	xcomm = new QDockWidget(this);
	xcomm->setWindowTitle("Command Line");
	QLineEdit* LE_Comm = new xLineEditWidget;
	xcomm->setWidget(LE_Comm);
	//connect(LE_Comm, SIGNAL(up_arrow_key_press()), this, SLOT(write_command_line_passed_data()));
	connect(LE_Comm, SIGNAL(editingFinished()), this, SLOT(xEditCommandLine()));
	//layout_comm->addWidget(LE_Comm);
	xcl = new xCommandLine;
	addDockWidget(Qt::TopDockWidgetArea, xcomm);
	ui.xIrrchlitArea->setWidget(xgl);
	QMainWindow::show();

	addDockWidget(Qt::BottomDockWidgetArea, xcw);
	this->setWindowState(Qt::WindowState::WindowMaximized);
	setupMainOperations();
	setupObjectOperations();
	setupShorcutOperations();
	xnavi = new xModelNavigator(this);
	addDockWidget(Qt::LeftDockWidgetArea, xnavi);
	setAcceptDrops(true);
	connect(xnavi, SIGNAL(definedSimulationWidget(wsimulation*)), this, SLOT(xGetSimulationWidget(wsimulation*)));
	connect(xnavi, SIGNAL(definedPointMassWidget(wpointmass*)), this, SLOT(xGetPointMassWidget(wpointmass*)));
	connect(xnavi, SIGNAL(InitializeWidgetStatement()), this, SLOT(xInitializeWidgetStatement()));
	connect(xgl, SIGNAL(signalGeometrySelection(QString)), this, SLOT(xGeometrySelection(QString)));
	connect(xgl, SIGNAL(releaseOperation()), this, SLOT(xReleaseOperation()));
	xNew();
}

// void xdynamics_gui::xInitializeGUI()
// {
// 	
// }

void xdynamics_gui::xGetSimulationWidget(wsimulation* w)
{
	wsim = w;
	connect(w, SIGNAL(clickedSolveButton(double, unsigned int, double)), this, SLOT(xRunSimulationThread(double, unsigned int, double)));
}

void xdynamics_gui::xGetPointMassWidget(wpointmass* w)
{
	wpm = w;
	QString n = wpm->LEName->text();
	xPointMass* xpm = NULL;
	if (xdm)
		if (xdm->XMBDModel())
			xpm = xdm->XMBDModel()->XMass(n.toStdString());
	if (xpm)
	{
		wpm->UpdateInformation(xpm);
	}
	connect(w, SIGNAL(clickEnableConnectGeometry(bool)), this, SLOT(xOnGeometrySelectionOfPointMass()));
}

xdynamics_gui::~xdynamics_gui()
{
	if (xgl) delete xgl; xgl = NULL;
	if (xcw) delete xcw; xcw = NULL;
	if (xnavi) delete xnavi; xnavi = NULL;
	if (xdm) delete xdm; xdm = NULL;
	if (xcomm) delete xcomm; xcomm = NULL;
	if (xcl) delete xcl; xcl = NULL;
	if (xchart) delete xchart; xchart = NULL;
	xvAnimationController::releaseTimeMemory();
}

void xdynamics_gui::Clear()
{
	xgl->ClearViewObject();
	xnavi->ClearTreeObject();
	xcw->ClearCommandText();
	if (xdm) delete xdm; xdm = NULL;
	isOnViewModel = false;
}

bool xdynamics_gui::ReadViewModel(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	if (!qf.isOpen())
		return false;
	int ver = 0;
	bool isExistParticle = false;
	xViewObjectType vot;
	int ns = 0;
	QString name;
	qf.read((char*)&ver, sizeof(int));
	while (!qf.atEnd())
	{
		qf.read((char*)&vot, sizeof(xViewObjectType));
		qf.read((char*)&ns, sizeof(int));
		char* _name = new char[255];
		memset(_name, 0, sizeof(char) * 255);
		qf.read((char*)_name, sizeof(char) * ns);
		name.sprintf("%s", _name);
		if (vot == xViewObjectType::VPLANE)
		{
			xPlaneObjectData d = { 0, };
			qf.read((char*)&d, sizeof(xPlaneObjectData));
			xgl->createPlaneGeometry(name, d);
			xnavi->addChild(xModelNavigator::SHAPE_ROOT, name);
		}
		else if (vot == xViewObjectType::VCUBE)
		{
			xCubeObjectData d = { 0, };
			qf.read((char*)&d, sizeof(xCubeObjectData));
			xgl->createCubeGeometry(name, d);
			xnavi->addChild(xModelNavigator::SHAPE_ROOT, name);
		}
		else if (vot == xViewObjectType::VMARKER)
		{
			xPointMassData d = { 0, };
			qf.read((char*)&d, sizeof(xPointMassData));
			QString marker_name = name + "_marker";
			xvMarker* xvm = xgl->makeMarker(marker_name, d);
			xvObject* xvo = xgl->Object(name);
			if (xvo)
			{
				xvo->setConnectedMassName(name);
				if (xdm)
				{
					if (xdm->XMBDModel())
					{
						xPointMass* xpm = xdm->XMBDModel()->XMass(name.toStdString());
						if (xpm)
							xpm->setConnectedGeometryName(xvo->Name());
					}
				}
			}
			xnavi->addChild(xModelNavigator::MASS_ROOT, name);
		}
		else if (vot == xViewObjectType::VPARTICLE)
		{
			xgl->vParticles()->defineFromViewFile(name);
			QStringList qsl = xgl->vParticles()->ParticleGroupData().keys();
			xnavi->addChilds(xModelNavigator::PARTICLE_ROOT, qsl);
			isExistParticle = true;
		}
		else if (vot == xViewObjectType::VJOINT)
		{
			xJointData d = { 0, };
			qf.read((char*)&d, sizeof(xJointData));
		}
		else if (vot == xViewObjectType::VTSDA)
		{
			xTSDAData d = { 0, };
			qf.read((char*)&d, sizeof(xTSDAData));
		}
		else if (vot == xViewObjectType::VRAXIAL)
		{
			xRotationalAxialForceData d = { 0, };
			qf.read((char*)&d, sizeof(xRotationalAxialForceData));
		}
		else if (vot == xViewObjectType::VMESH)
		{
			xgl->createMeshObjectGeometry(name);
		}
		delete[] _name;
	}
	qf.close();
	xgl->fitView();
	return true;
}

bool xdynamics_gui::ReadModelResults(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	QTextStream qts(&qf);
	QString s;
	qts >> s;
	if (s == "MBD")
	{

	}
	else if (s == "DEM")
	{
		QStringList sl;
		while (!qts.atEnd())
		{
			qts >> s;
			if (!s.isEmpty())
				sl.push_back(s);
		}
		if(!xgl->Upload_DEM_Results(sl))
		{
			return false;
		}
	}
	qf.close();
	return true;
}

void xdynamics_gui::xNew()
{
	xNewDialog nd(NULL, path);
	int ret = nd.exec();
	if (ret)
	{
		path = nd.path;
		Clear();
		//ClearMemory();
		//xInitializeGUI();
		if (nd.isBrowser)
			OpenFile(nd.pathinbrowser);
		else
		{

		}
	}
}

void xdynamics_gui::xSave()
{

}

void xdynamics_gui::xOpen()
{
	QString file_path = QFileDialog::getOpenFileName(
		this, tr("open"), path,
		tr("View model file (*.vmd);;Ecxel File(*.xls);;All files(*.*)"));
	if (!file_path.isEmpty())
		OpenFile(file_path);
}

void xdynamics_gui::xCylinder()
{
	caction = CYLINDER;
	xcomm->widget()->setFocus();
	xcomm->setWindowTitle("Input the top point.");
	xcl->SetCurrentAction(caction);
}

void xdynamics_gui::xCube()
{
	caction = CUBE;
	xcomm->widget()->setFocus();
	xcomm->setWindowTitle("Input the minimum point.");
	xcl->SetCurrentAction(caction);
}

void xdynamics_gui::xChart()
{
	caction = CHART;
	if (!xchart)
	{
		xchart = new xChartWindow(this);
		xchart->setChartData(xdm);
		xchart->show();
		return;
	}
	if (xchart->isVisible())
		return;
	else
	{
		delete xchart;
		xchart = NULL;
		xChart();
	}
}

void xdynamics_gui::OpenFile(QString s)
{
	int begin = s.lastIndexOf('.');
	QString ext = s.mid(begin + 1);
	xcw->write(xCommandWindow::CMD_INFO, "File load - " + s);
	if (ext == "vmd")
	{
		if (isOnViewModel)
		{
			Clear();
		}
		isOnViewModel = ReadViewModel(s);
	}		
	else if (ext == "rlt")
	{
		if (!isOnViewModel)
		{
			QString p = s.left(begin) + ".vmd";
			if (!ReadViewModel(p))
			{
				xcw->write(xCommandWindow::CMD_ERROR, kor("해석 결과에 부합하는 모델을 찾을 수 없습니다."));
				return;
			}
			else isOnViewModel = true;
		}
		if (ReadModelResults(s)) setupAnimationTool();
	}
	else if (ext == "xls")
	{
		if (isOnViewModel)
		{
			Clear();
		}
		if (!isOnViewModel)
		{
			QString vf = ReadXLSFile(s);
 			if (!vf.isEmpty())
 				isOnViewModel = ReadViewModel(vf);
		}
	}
	else
		xcw->write(xCommandWindow::CMD_ERROR, kor("지원하지 않는 파일 형식입니다."));
}

QString xdynamics_gui::ReadXLSFile(QString xls_path)
{
	if (!xdm)
	{
		xdm = new xDynamicsManager;
		//xnavi->setDynamicManager(xdm);
	}		
	xdm->OpenModelXLS(xls_path.toStdWString().c_str());
	int begin = xls_path.lastIndexOf("/");
	int end = xls_path.lastIndexOf(".");
	QString modelName = xls_path.mid(begin + 1, end - begin - 1);
	QString viewFile = path + modelName + "/" + modelName + ".vmd";
	return viewFile;
}

void xdynamics_gui::setupMainOperations()
{
	QAction* a;

	ui.mainToolBar->setWindowTitle("Main Operations");

	a = new QAction(QIcon(":/Resources/icon/new.png"), tr("&New"), this);
	a->setStatusTip(tr("New"));
	connect(a, SIGNAL(triggered()), this, SLOT(xNew()));
	myMainActions.insert(NEW, a);

	a = new QAction(QIcon(":/Resources/icon/open.png"), tr("&Open"), this);
	a->setStatusTip(tr("Open project"));
	connect(a, SIGNAL(triggered()), this, SLOT(xOpen()));
	myMainActions.insert(OPEN, a);

	a = new QAction(QIcon(":/Resources/icon/save.png"), tr("&Save"), this);
	a->setStatusTip(tr("Save project"));
	connect(a, SIGNAL(triggered()), this, SLOT(xSave()));
	myMainActions.insert(SAVE, a);

	for (int i = 0; i < myMainActions.size(); i++)
	{
		ui.mainToolBar->addAction(myMainActions.at(i));
	}
	/*connect(ui.Menu_import, SIGNAL(triggered()), this, SLOT(SHAPE_Import()));*/
}

void xdynamics_gui::setupObjectOperations()
{
	myObjectBar = addToolBar(tr("Object Operations"));
	QAction* a;

	//ui.mainToolBar->setWindowTitle("Main Operations");

	a = new QAction(QIcon(":/Resources/icon/cube.png"), tr("&Cube"), this);
	a->setStatusTip(tr("Cube object"));
	connect(a, SIGNAL(triggered()), this, SLOT(xCube()));
	myObjectActions.insert(CUBE, a);

	a = new QAction(QIcon(":/Resources/icon/cylinder.png"), tr("&Cylinder"), this);
	a->setStatusTip(tr("Cylinder object"));
	connect(a, SIGNAL(triggered()), this, SLOT(xCylinder()));
	myObjectActions.insert(CYLINDER, a);

// 	a = new QAction(QIcon(":/Resources/icon/plot.png"), tr("&Chart"), this);
// 	a->setStatusTip(tr("Chart window"));
// 	connect(a, SIGNAL(triggered()), this, SLOT(xChart()));
// 	myObjectActions.insert(CHART, a);

// 	a = new QAction(QIcon(":/Resources/icon/save.png"), tr("&Save"), this);
// 	a->setStatusTip(tr("Save project"));
// 	connect(a, SIGNAL(triggered()), this, SLOT(xSave()));
// 	myMainActions.insert(SAVE, a);

	for (int i = 0; i < myObjectActions.size(); i++)
	{
		myObjectBar->addAction(myObjectActions.at(i));
	}
}

void xdynamics_gui::setupAnimationTool()
{
	if (!myAnimationBar)
	{
		myAnimationBar = new xAnimationTool(this);
		myAnimationBar->setup(xgl);
		this->addToolBar(myAnimationBar);
		//connect(xgl, SIGNAL(changedAnimationFrame()), this, SLOT(xChangeAnimationFrame()));
		//this->insertToolBar(ui.mainToolBar, myAnimationBar);
	}
}

void xdynamics_gui::setupBindingPointer()
{
	if (sThread)
	{
		if (xdm)
		{
// 			QString n = "crank";
// 			xPointMass* pm = xdm->XMBDModel()->XMass(n.toStdString());
// 			xvObject* obj = xgl->Object(n + "d");
// 			obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
			if (xdm->XMBDModel())
			{
				foreach(xPointMass* pm, xdm->XMBDModel()->Masses())
				{
					QString n = pm->ConnectedGeometryName();
					xvObject* obj = xgl->Object(n);
					if(obj)
						obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
					obj = xgl->Object(pm->Name() + "_marker");
					if (obj)
						obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
				}
			}			
		}
	}
}

void xdynamics_gui::setupShorcutOperations()
{
	QShortcut *a = new QShortcut(QKeySequence("Ctrl+Q"), this);
	connect(a, SIGNAL(activated()), this, SLOT(xStopSimulationThread()));
}

void xdynamics_gui::dragEnterEvent(QDragEnterEvent *event)
{
	event->acceptProposedAction();
}

// void xdynamics_gui::dragMoveEvent(QDragMoveEvent *event)
// {
// 
// }

void xdynamics_gui::dropEvent(QDropEvent *event)
{
	const QMimeData* mimeData = event->mimeData();
	QStringList pathList;
	if (mimeData->hasUrls())
	{		
		QList<QUrl> urlList = mimeData->urls();
		for (unsigned int i = 0; i < urlList.size(); i++)
		{
			pathList.append(urlList.at(i).toLocalFile());
		}
	}
	foreach(QString s, pathList)
	{
		OpenFile(s);
	}
}

void xdynamics_gui::xExitSimulationThread()
{
	//onAnimationPause();
	sThread->quit();
	sThread->wait();
	sThread->disconnect();
	if (sThread) delete sThread; sThread = NULL;
	if (pbar)
	{
		delete pbar;
		pbar = NULL;
	}
	//errors::Error(model::name);
}

void xdynamics_gui::xRecieveProgress(int pt, QString ch)
{
	if (pt >= 0 && !ch.isEmpty())
	{
 		myAnimationBar->update(pt);
// 		//myAnimationBar->AnimationSlider()->setMaximum(pt);
 		 		
 		xcw->write(xCommandWindow::CMD_INFO, ch);
 		if (xgl->vParticles())
 		{
 			QString fileName;
			fileName.sprintf("Part%04d", pt);
 			xgl->vParticles()->UploadParticleFromFile(pt, path + xModel::name + "/" + fileName + ".bin");
 		}
 		xvAnimationController::setTotalFrame(pt);
	}
	else if (pt == -1 && !ch.isEmpty())
	{
		xcw->write(xCommandWindow::CMD_INFO, ch);
	}
	else if (pt == ERROR_DETECTED)
	{
		xcw->write(xCommandWindow::CMD_ERROR, ch);
		//xExitSimulationThread();
	}
	else if (pt >= 0 && ch.isEmpty())
	{
		pbar->setValue(pt);
	}
}

void xdynamics_gui::xEditCommandLine()
{
	QLineEdit* e = (QLineEdit*)sender();
	QString c = e->text();
	switch (caction)
	{
	case CYLINDER:
	{
		QString msg = xcl->CylinderCommandProcess(c);
		if (xcl->IsWrongCommand())
		{
			xcw->write(xCommandWindow::CMD_INFO, msg);
		}
		xcomm->setWindowTitle(msg);
		if (xcl->IsFinished())
		{
			xCylinderObjectData d = xcl->GetCylinderParameters();
			QString n = QString("Cylinder%1").arg(xvObject::xvObjectCount());
			xgl->createCylinderGeometry(n, d);
			xnavi->addChild(xModelNavigator::SHAPE_ROOT, n);
			caction = -1;
		}
		e->clear();
		break;
	}
	case CUBE:
	{
		QString msg = xcl->CubeCommandProcess(c);
		if (xcl->IsWrongCommand())
		{
			xcw->write(xCommandWindow::CMD_INFO, msg);
		}
		xcomm->setWindowTitle(msg);
		if (xcl->IsFinished())
		{
			xCubeObjectData d = xcl->GetCubeParameters();
			QString n = QString("Cube%1").arg(xvObject::xvObjectCount());
			xgl->createCubeGeometry(n, d);
			xnavi->addChild(xModelNavigator::SHAPE_ROOT, n);
			caction = -1;
		}
		e->clear();
		break;
	}
	default:
		break;
	}
}

void xdynamics_gui::xGeometrySelection(QString n)
{
	if (wpm)
	{
		if (wpm->IsOnConnectGeomegry())
		{
			if (xdm)
			{
				if (xdm->XMBDModel())
				{
					xPointMass* pm = xdm->XMBDModel()->XMass(wpm->LEName->text().toStdString());
					if (pm)
					{
						//QString n = pm->Name();
						xvObject* obj = xgl->Object(n);
						obj->setConnectedMassName(pm->Name());
						pm->setConnectedGeometryName(obj->Name());
						if (pm->XPointMassResultPointer())
							obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
						wpm->LEGeometry->setText(obj->Name());
						xcw->write(xCommandWindow::CMD_INFO, "The geometry(" + obj->Name() + ") is connected to point mass(" + pm->Name() + ").");
					}
				}
			}
		}
		wpm->IsOnConnectGeomegry() = false;
	}
}

void xdynamics_gui::xReleaseOperation()
{
	if (wpm)
		wpm->IsOnConnectGeomegry() = false;
	ui.statusBar->setStatusTip(QString(""));
}

void xdynamics_gui::xInitializeWidgetStatement()
{
	if (wpm) wpm = NULL;
	if (wsim) wsim = NULL;
}

void xdynamics_gui::xOnGeometrySelectionOfPointMass()
{
	ui.statusBar->setStatusTip(QString("Select the geometry for connecting the point mass"));
}

void xdynamics_gui::xStopSimulationThread()
{
	if (sThread)
		sThread->setStopCondition();
}

void xdynamics_gui::deleteFileByEXT(QString ext)
{
	QString dDir = path + xModel::name;
	QDir dir = QDir(dDir);
	QStringList delFileList;
	delFileList = dir.entryList(QStringList("*." + ext), QDir::Files | QDir::NoSymLinks);
	qDebug() << "The number of *.bin file : " << delFileList.length();
	for (int i = 0; i < delFileList.length(); i++){
		QString deleteFilePath = dDir + "/" + delFileList[i];
		QFile::remove(deleteFilePath);
	}
	qDebug() << "Complete delete.";
}

void xdynamics_gui::xRunSimulationThread(double dt, unsigned int st, double et)
{
	deleteFileByEXT("txt");
	deleteFileByEXT("bin");
	setupAnimationTool();
	sThread = new xSimulationThread;
	sThread->xInitialize(xdm, dt, st, et);
	//xcw->write(xCommandWindow::CMD_INFO, QString("%1, %1, %1").arg(dt).arg(st).arg(et));
	//xcw->write(xCommandWindow::CMD_INFO, "Thread Initialize Done.");
	xgl->vParticles()->setBufferMemories(xSimulation::npart);
	connect(sThread, SIGNAL(finishedThread()), this, SLOT(xExitSimulationThread()));
	connect(sThread, SIGNAL(sendProgress(int, QString)), this, SLOT(xRecieveProgress(int, QString)));
	//xcw->write(xCommandWindow::CMD_INFO, "Thread Initialize Done.");
	if (!pbar) pbar = new QProgressBar;
	//xcw->write(xCommandWindow::CMD_INFO, "Thread Initialize Done.");
	//unsigned int nstep = xSimulation::nstep;
	pbar->setMaximum(xSimulation::nstep);
	ui.statusBar->addWidget(pbar, 1);
	setupBindingPointer();
	sThread->start();
	
	//xcw->write(xCommandWindow::CMD_INFO, "Thread Initialize Done.");
// 	solveDialog sd;
// 	int ret = sd.exec();
// 	if (ret <= 0)
// 		return;
// 	if (model::isSinglePrecision && sd.isCpu)
// 	{
// 		messageBox::run("Single precision does NOT provide the CPU processing.");
// 		return;
// 	}
// 	simulation::dt = sd.time_step;
// 	simulation::et = sd.sim_time;
// 	simulation::st = sd.save_step;
// 	simulation::dev = sd.isCpu ? simulation::CPU : simulation::GPU;
// 
// 	deleteFileByEXT("txt");
// 	deleteFileByEXT("bin");
// 	if (!solver)
// 		solver = new xDynamicsSolver(mg);
// 	connect(solver, SIGNAL(finishedThread()), this, SLOT(exitThread()));
// 	connect(solver, SIGNAL(excuteMessageBox()), this, SLOT(excuteMessageBox()));
// 	connect(solver, SIGNAL(sendProgress(int, QString, QString)), this, SLOT(recieveProgress(int, QString, QString)));
// 	if (solver->initialize(sd.dem_itor_type, sd.mbd_itor_type, st_model))
// 	{
// 
// 	}
// 	else
// 	{
// 		exitThread();
// 	}
// 	if (!pBar)
// 		pBar = new QProgressBar;
// 	pBar->setMaximum(solver->totalPart());
// 	//
// 	ui.statusBar->addWidget(pBar, 1);
// 	myModelingActions[RUN_ANALYSIS]->disconnect();
// 	myModelingActions[RUN_ANALYSIS]->setIcon(QIcon(":/Resources/stop.png"));
// 	myModelingActions[RUN_ANALYSIS]->setStatusTip(tr("Pause for simulation."));
// 	connect(myModelingActions[RUN_ANALYSIS], SIGNAL(triggered()), solver, SLOT(setStopCondition()));
// 	myModelingActions[RUN_ANALYSIS]->setEnabled(true);
// 	saveproj();
// 	solver->start();
// 	setAnimationAction(true);
}