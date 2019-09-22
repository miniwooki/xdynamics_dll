#include "xdynamics_gui.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include "xModelNavigator.h"
#include "xSimulationThread.h"
#include "xResultCallThread.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xNewDialog.h"
#include "xLineEditWidget.h"
#include "xListWidget.h"
#include "xpass_distribution_dlg.h"
#include "xCommandLine.h"
#include "xChartWindow.h"
#include "xColorControl.h"
#include <QtCore/QDir>
#include <QtCore/QMimeData>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QShortcut>
//#include <QtWidgets/QListWidget>

xdynamics_gui* xgui;
wsimulation* wsim;
wresult* wrst = NULL;
wpointmass* wpm;
static xSimulationThread* sThread = NULL;
static xResultCallThread* rThread = NULL;

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
	, xcc(NULL)
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
	connect(xnavi, SIGNAL(definedResultWidget(wresult*)), this, SLOT(xGetResultWidget(wresult*)));
	connect(xnavi, SIGNAL(InitializeWidgetStatement()), this, SLOT(xInitializeWidgetStatement()));
	connect(xgl, SIGNAL(signalGeometrySelection(QString)), this, SLOT(xGeometrySelection(QString)));
	connect(xgl, SIGNAL(releaseOperation()), this, SLOT(xReleaseOperation()));
	connect(xgl, SIGNAL(contextSignal(QString, contextMenuType)), this, SLOT(xContextMenuProcess(QString, contextMenuType)));
	RECT desktop;
	const HWND hDesktop = GetDesktopWindow();
	GetWindowRect(hDesktop, &desktop);
	desktop_size.setWidth(desktop.right);
	desktop_size.setHeight(desktop.bottom);
	xNew();
}

void xdynamics_gui::xGetSimulationWidget(wsimulation* w)
{
	wsim = w;
	connect(w, SIGNAL(clickedSolveButton(double, unsigned int, double)), this, SLOT(xRunSimulationThread(double, unsigned int, double)));
	connect(w, SIGNAL(clickedStartPointButton()), this, SLOT(xSelectStartPoint()));
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

void xdynamics_gui::xGetResultWidget(wresult *w)
{
	wrst = w;
	if (xdm->XDEMModel())
	{
		connect(w, SIGNAL(clickedApplyButton(int)), this, SLOT(xSetupParticleBufferColorDistribution(int)));
		connect(w, SIGNAL(changedTargetCombo(int)), this, SLOT(xSetupResultNavigatorByChangeTargetCombo(int)));
	}
	
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
	if (xcc) delete xcc; xcc = NULL;
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
			//xvm->setMarkerScale(0.1);
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
						{
							xpm->setConnectedGeometryName(xvo->Name().toStdString());							
							vector3d p = xpm->Position();
							xvo->setPosition(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));								
						}
					}
				}
			}
			xnavi->addChild(xModelNavigator::MASS_ROOT, name);
		}
		else if (vot == xViewObjectType::VPARTICLE)
		{
			if (!xgl->vParticles())
				xgl->createParticles();
			if (xgl->vParticles())
			{
				xgl->vParticles()->defineFromViewFile(name);
				QStringList qsl = xgl->vParticles()->ParticleGroupData().keys();
				xnavi->addChilds(xModelNavigator::PARTICLE_ROOT, qsl);
				isExistParticle = true;
			}
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
			xvMeshObject* xm = xgl->createMeshObjectGeometry(name);			
		}
		else if (vot == xViewObjectType::VCYLINDER)
		{
			xCylinderObjectData d = { 0, };
			qf.read((char*)&d, sizeof(xCylinderObjectData));
			xgl->createCylinderGeometry(name, d);
			xnavi->addChild(xModelNavigator::SHAPE_ROOT, name);
		}
		delete[] _name;
	}
	qf.close();
	
	xgl->fitView();
	return true;
}

bool xdynamics_gui::ReadModelResults(QString path)
{
	rThread = new xResultCallThread;
	rThread->set_dynamics_manager(xdm, path);
	setupBindingPointer();
	connect(rThread, SIGNAL(result_call_finish()), this, SLOT(xReleaseResultCallThread()));
	connect(rThread, SIGNAL(result_call_send_progress(int, QString)), this, SLOT(xRecieveProgress(int, QString)));

	if (!pbar) pbar = new QProgressBar;
	pbar->setMaximum(xdm->XResult()->get_num_parts() + 10);
	ui.statusBar->addWidget(pbar, 1);
	rThread->start();
	/*if (xdm)
	{
		if (xdm->upload_model_results(path.toStdString()))
		{
			xvAnimationController::allocTimeMemory(xdm->XResult()->get_num_parts());
			double* _time = xdm->XResult()->get_times();
			for (unsigned int i = 0; i < xdm->XResult()->get_num_parts(); i++)
				xvAnimationController::addTime(i, static_cast<float>(_time[i]));
			setupBindingPointer();
			xvAnimationController::setTotalFrame(xdm->XResult()->get_num_parts());
		}
	}*/
	//QFile qf(path);
	//qf.open(QIODevice::ReadOnly);
	//QTextStream qts(&qf);
	//QString s;
	//QString dyn_type = "";
	//double dt = 0.0;
	//unsigned int st = 0;
	//double et = 0;
	//unsigned int pt = 0;
	//if (!pbar) pbar = new QProgressBar;
	//qts >> s;
	//if (s == "SIMULATION")
	//{
	//	qts >> dt >> st >> et >> pt;
	//	xvAnimationController::allocTimeMemory(pt);
	//	for (unsigned int i = 0; i < pt; i++)
	//	{
	//		xvAnimationController::addTime(i, i * dt);
	//	}
	//}
	//pbar->setMaximum(pt);
	//QStringList mbd_rlist;
	//QStringList dem_rlist;
	//while (!qts.atEnd())
	//{
	//	qts >> s;
	//	if (s == "MBD" || s == "DEM")
	//	{
	//		dyn_type = s;
	//		if (dyn_type == "DEM")
	//		{
	//			if (xgl->vParticles())
	//			{
	//				xgl->vParticles()->setBufferMemories(pt);
	//				xnavi->addChild(xModelNavigator::RESULT_ROOT, "Particles");
	//			}
	//		}
	//		continue;
	//	}
	//	if (dyn_type == "MBD")
	//		mbd_rlist.push_back(s);
	//	else if (dyn_type == "DEM")
	//		dem_rlist.push_back(s);
	//}
	//pbar->setMaximum(mbd_rlist.size() + dem_rlist.size());
	//unsigned int cnt = 0;
	//foreach(QString f, mbd_rlist)
	//{
	//	pbar->setValue(cnt++);
	//	QString file_name = QString::fromStdString(xUtilityFunctions::GetFileName(f.toStdString().c_str()));
	//	QString ext = QString::fromStdString(xUtilityFunctions::FileExtension(f.toStdString().c_str()));
	//	if (ext == ".bpm")
	//	{
	//		xPointMass* xpm = NULL;
	//		if (xdm->XMBDModel())
	//			xpm = xdm->XMBDModel()->XMass(file_name.toStdString());
	//		if (xpm)
	//		{
	//			xpm->ImportResults(f.toStdString());
	//			xcw->write(xCommandWindow::CMD_INFO, "Imported file : " + f);
	//		}
	//	}
	//	if (ext == ".bkc")
	//	{
	//		xKinematicConstraint* xkc = NULL;
	//		if (xdm->XMBDModel())
	//			xkc = xdm->XMBDModel()->XJoint(file_name.toStdString());
	//		if (xkc)
	//		{
	//			xkc->ImportResults(f.toStdString());
	//			xcw->write(xCommandWindow::CMD_INFO, "Imported file : " + f);
	//		}
	//		else
	//		{
	//			xDrivingConstraint* xdc = xdm->XMBDModel()->xDriving(file_name.toStdString());
	//			if (xdc)
	//			{
	//				xdc->ImportResults(f.toStdString());
	//				xcw->write(xCommandWindow::CMD_INFO, "Imported file : " + f);
	//			}
	//		}
	//	}
	//}
	//unsigned int dem_cnt = 0;
	//foreach(QString f, dem_rlist)
	//{
	//	pbar->setValue(cnt++);
	//	xgl->vParticles()->UploadParticleFromFile(dem_cnt, f);
	//	//	xvAnimationController::setTotalFrame(pt);
	//	xgl->setupParticleBufferColorDistribution(dem_cnt++);
	//	xcw->write(xCommandWindow::CMD_INFO, "Imported file : " + f);
	//}
	//if (wrst)
	//{
	//	wrst->setMinMaxValue(
	//		xgl->GetParticleMinValueFromColorMapType(),
	//		xgl->GetParticleMaxValueFromColorMapType()
	//	);
	//}

	//setupBindingPointer();
	//xvAnimationController::setTotalFrame(pt - 1);
	//qf.close();
	//if (pbar)
	//{
	//	delete pbar;
	//	pbar = NULL;
	//}
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
		xResultManager* _xrm = xdm->XResult();
		if (!_xrm)
		{
			xcw->write(xCommandWindow::CMD_INFO, "The analysis result does not exist.");
			return;
		}			
		xchart = new xChartWindow(this);
		xchart->setChartData(xdm->XResult());
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
				xcw->write(xCommandWindow::CMD_ERROR, "Can't find the model relate with analysis results.");// 해석 결과에 부합하는 모델을 찾을 수 없습니다.");
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
	else if (ext == "stl")
	{
		xgl->ReadSTLFile(s);
	}
	else
		xcw->write(xCommandWindow::CMD_ERROR, "Unsupported file format.");
}

QSize xdynamics_gui::FullWindowSize()
{
	return desktop_size;
}

QString xdynamics_gui::ReadXLSFile(QString xls_path)
{
	if (!xdm)
	{
		xdm = new xDynamicsManager;
		//xnavi->setDynamicManager(xdm);
	}		
	if (checkXerror(xdm->OpenModelXLS(xls_path.toStdString().c_str())))
	{
		delete xdm;
		xdm = NULL;
		xcw->write(xCommandWindow::CMD_ERROR, "Error in xls file\nCheck log file.");
		return "";
	}
	int begin = xls_path.lastIndexOf("/");
	int end = xls_path.lastIndexOf(".");
	QString modelName = xls_path.mid(begin + 1, end - begin - 1);
	QString viewFile = path + modelName + "/" + modelName + ".vmd";
	return viewFile;
}

void xdynamics_gui::ReadSTLFile(QString path)
{

}

void xdynamics_gui::setupMeshSphere()
{
	foreach(xvObject* xvo, xgl->Objects())
	{
		if (xvo->ObjectType() == xvObject::V_POLYGON)
		{
			std::string nm = xvo->Name().toStdString();
			std::string m_path = xModel::makeFilePath(nm) + ".tsd";
			bool b = xUtilityFunctions::ExistFile(m_path.c_str());
			if (b)
			{
				QFile qf(m_path.c_str());
				qf.open(QIODevice::ReadOnly);
				unsigned int sz = 0;
				qf.read((char*)&sz, sizeof(unsigned int));
				double *data = new double[sz * 3];
				double *rdata = new double[sz];
				qf.read((char*)data, sizeof(double) * sz * 3);
				qf.read((char*)rdata, sizeof(double) * sz);
				qf.close();
				xvParticle *xp = xgl->createParticleObject(xvo->Name());
				xp->setRelativePosition(sz, data, rdata);
				xObject* xo = xdm->XObject()->XObject(nm);
				if (xo)
				{
					xPointMass* xpm = dynamic_cast<xPointMass*>(xo);
					xp->defineFromRelativePosition(xpm->Position(), xpm->EulerParameters());
				}
				delete[] data;
				delete[] rdata;
			}
		}
	}	
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

	a = new QAction(QIcon(":/Resources/icon/plot.png"), tr("&Chart"), this);
	a->setStatusTip(tr("Chart window"));
	connect(a, SIGNAL(triggered()), this, SLOT(xChart()));
	myObjectActions.insert(CHART, a);

	a = new QAction(QIcon(":/Resources/icon/upload_result.png"), tr("&Uplaod_Result"), this);
	a->setStatusTip(tr("Uplaod results"));
	connect(a, SIGNAL(triggered()), this, SLOT(xUploadResultThisModel()));
	myObjectActions.insert(UPLOAD_RESULT, a);

	a = new QAction(QIcon(":/Resources/icon/chart_passing.png"), tr("&Passing distribution"), this);
	a->setStatusTip(tr("Distribution of particles passing area"));
	connect(a, SIGNAL(triggered()), this, SLOT(xPassFiyribution()));
	myObjectActions.insert(PASSING_DISTRIBUTION, a);
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
	if (xdm)
	{
		if (!xcc) 
			xcc = new xColorControl;
		xResultManager* xrm = xdm->XResult();
		if (xdm->XMBDModel())
		{
			for (xmap<xstring, xPointMass*>::iterator it = xdm->XMBDModel()->Masses().begin(); it != xdm->XMBDModel()->Masses().end(); it.next())
			//foreach(xPointMass* pm, xdm->XMBDModel()->Masses())
			{
				xPointMass* pm = it.value();
				QString n = QString::fromStdString(pm->Name());
				xvObject* obj = xgl->Object(n);
				xPointMass::pointmass_result* pmrs = NULL;
				pmrs = xrm->get_mass_result_ptr(n.toStdString());
				if (obj)
					if(pmrs)
						obj->bindPointMassResultsPointer(pmrs);
					
				obj = xgl->Object(QString::fromStdString(pm->Name()) + "_marker");
				if (obj && pmrs)
					obj->bindPointMassResultsPointer(pmrs);
			}
		}
		else
		{
		/*	foreach(xObject* xo, xdm->XObject()->CompulsionMovingObjects())
			{
				xPointMass* pm = dynamic_cast<xPointMass*>(xo);
				xvObject* obj = xgl->Object(xo->Name());
				if (obj)
					obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
				obj = xgl->Object(xo->Name() + "_marker");
				if (obj)
					obj->bindPointMassResultsPointer(pm->XPointMassResultPointer());
			}*/
		}
		if (xdm->XDEMModel())
		{
			xgl->vParticles()->bind_result_buffers(
				xrm->get_particle_position_result_ptr(),
				xrm->get_particle_velocity_result_ptr(),
				xrm->get_particle_color_result_ptr());
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
 		xcw->write(xCommandWindow::CMD_INFO, ch);
		xdm->XResult()->setup_particle_buffer_color_distribution(xcc, pt, pt);
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

void xdynamics_gui::xSetupResultNavigatorByChangeTargetCombo(int cmt)
{
	if (wrst && xcc)
	{
		if (!xdm)
			return;
		xColorControl::setTarget((xColorControl::ColorMapType)cmt);
		if (!xColorControl::isUserLimitInput())
		{
			wrst->setMinMaxValue(
				xdm->XResult()->get_min_result_value(xcc->Target()),
				xdm->XResult()->get_max_result_value(xcc->Target())
			);
		}
	}
}

void xdynamics_gui::xUploadResultThisModel()
{
	//QFile qf;
	std::string path = xModel::makeFilePath("");//QString::fromStdString("");
//	qf.setFileName(path);
	//qf.open(QIODevice::ReadOnly);
	if (!isOnViewModel)
	{
		xcw->write(xCommandWindow::CMD_ERROR, "No model is defined that matches the result.");
		return;
	}
	if (ReadModelResults(QString::fromStdString(path))) setupAnimationTool();
}

void xdynamics_gui::xSetupParticleBufferColorDistribution(int ef)
{
	if (xdm && xcc)
	{
		xdm->XResult()->setup_particle_buffer_color_distribution(xcc, 0, ef);
	}
}

void xdynamics_gui::xSelectStartPoint()
{
	xListWidget lw(this);// = new xListWidget;
	if (result_file_list.size())
	{
		lw.setup_widget(result_file_list);//lw->addItems(result_file_list);
		int ret = lw.exec();
		if (ret)
		{
			QString sitem = lw.get_selected_item();

			int begin = sitem.lastIndexOf("/");
			QString mm = sitem.mid(begin + 5, 4);
			//last_pt = mm.toUInt();
			wsim->set_starting_point(sitem, mm.toUInt());
		}
	}
}

void xdynamics_gui::xPassDistribution()
{
	if (xdm->XResult())
	{
		xpass_distribution_dlg d(this);
		d.setup(xdm->XResult());
		int ret = d.exec();
		if (ret)
		{
			xdm->XResult()->set_distribution_result(d.get_distribution_result().toStdList());
		}
	}
	
}

void xdynamics_gui::xEditCommandLine()
{
	QLineEdit* e = (QLineEdit*)sender();
	QString c = e->text();
	switch (caction)
	{
	case CONVERT_MESH_TO_SPHERE:
	{
		double ft = c.toDouble();
		if (ft)
		{
			xvParticle* xp = xgl->createParticleObject(xcl->GetCurrentObject()->Name());
			//if (!xgl->vParticles())
				//xgl->createParticles();
			//if (xgl->vParticles())
			if(xp)
			{
				QString file = xcl->MeshObjectCommandProcess(c);
				//xgl->vParticles()->defineFromListFile(file);
				xp->defineFromListFile(file);
			}
		}		
		e->clear();
		break;
	}
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
						obj->setConnectedMassName(QString::fromStdString(pm->Name()));
						pm->setConnectedGeometryName(obj->Name().toStdString());
						xPointMass::pointmass_result* pmrs = NULL;
						pmrs = xdm->XResult()->get_mass_result_ptr(n.toStdString());
						if (pmrs)
							obj->bindPointMassResultsPointer(pmrs);
						wpm->LEGeometry->setText(obj->Name());
						xcw->write(xCommandWindow::CMD_INFO, "The geometry(" + obj->Name() + ") is connected to point mass(" + QString::fromStdString(pm->Name()) + ").");
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
	xgl->releaseSelectedObjects();
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

void xdynamics_gui::xReleaseResultCallThread()
{
	result_file_list = rThread->get_file_list();
	xcw->write(xCommandWindow::CMD_INFO, "The loading process of analysis results has been terminated.");
	myAnimationBar->update(xdm->XResult()->get_num_parts() - 1);
	rThread->quit();
	rThread->wait();
	rThread->disconnect();
	if (rThread) delete rThread; rThread = NULL;
	if (pbar)
	{
		delete pbar;
		pbar = NULL;
	}
	
}

void xdynamics_gui::xContextMenuProcess(QString nm, contextMenuType vot)
{
	switch (vot)
	{
	case CONTEXT_CONVERT_SPHERE:
		caction = CONVERT_MESH_TO_SPHERE;
		xcomm->widget()->setFocus();
		xcomm->setWindowTitle("Input the fit ratio.");
		xcl->SetCurrentAction(caction);
		xcl->SetCurrentObject(xgl->Objects()[nm]);
		break;
	}
}

void xdynamics_gui::deleteFileByEXT(QString ext)
{
	QString dDir = path + QString::fromStdString(xModel::name.toStdString());
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
	sThread = new xSimulationThread;
	
	if(wsim->get_enable_starting_point() == false)
	{
		deleteFileByEXT("txt");
		deleteFileByEXT("bin");		
		xvAnimationController::allocTimeMemory(xSimulation::npart);
	}
	sThread->xInitialize(xdm, dt, st, et, wsim->get_starting_part());
	if (wsim && wsim->get_enable_starting_point())
	{
		QString spath = wsim->get_starting_point_path();
		if (!spath.isEmpty())
		{
			sThread->set_from_part_result(spath);
		}
	}
	else
		setupBindingPointer();
	setupAnimationTool();
	
	if (xgl->vParticles())
		xnavi->addChild(xModelNavigator::RESULT_ROOT, "Particles");

	connect(sThread, SIGNAL(finishedThread()), this, SLOT(xExitSimulationThread()));
	connect(sThread, SIGNAL(sendProgress(int, QString)), this, SLOT(xRecieveProgress(int, QString)));
	if (!pbar) pbar = new QProgressBar;
	pbar->setMaximum(xSimulation::nstep);
	ui.statusBar->addWidget(pbar, 1);
	sThread->start();
}