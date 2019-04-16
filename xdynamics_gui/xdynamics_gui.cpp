#include "xdynamics_gui.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include <QtCore/QDir>
#include <QtCore/QMimeData>
#include <QtWidgets/QFileDialog>

xdynamics_gui::xdynamics_gui(int _argc, char** _argv, QWidget *parent)
	: QMainWindow(parent)
	, xgl(NULL)
	, xcw(NULL)
	, xnavi(NULL)
	, xdm(NULL)
	, isOnViewModel(false)
{
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
	ui.xIrrchlitArea->setWidget(xgl);
	QMainWindow::show();
	xcw = new xCommandWindow(this);
	addDockWidget(Qt::BottomDockWidgetArea, xcw);
	this->setWindowState(Qt::WindowState::WindowMaximized);
	setupMainOperations();
	xnavi = new xModelNavigator(this);
	addDockWidget(Qt::LeftDockWidgetArea, xnavi);
	setAcceptDrops(true);
// 	if (argc > 1)
// 	{
// 		QString model_name = argv[2];
// 		QString _path = path + model_name;
// 		if (!QDir(_path).exists())
// 		
// 		ReadViewModel(path);
// 	}
}


xdynamics_gui::~xdynamics_gui()
{
	if (xgl) delete xgl; xgl = NULL;
	if (xcw) delete xcw; xcw = NULL;
	if (xnavi) delete xnavi; xnavi = NULL;
	if (xdm) delete xdm; xdm = NULL;
	xvAnimationController::releaseTimeMemory();
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
		else if (vot == xViewObjectType::VPARTICLE)
		{
			xgl->vParticles()->defineFromViewFile(name);
			QStringList qsl = xgl->vParticles()->ParticleGroupData().keys();
			xnavi->addChilds(xModelNavigator::PARTICLE_ROOT, qsl);
			isExistParticle = true;
		}
		else if (vot == xViewObjectType::VMESH)
		{
			xgl->createMeshObjectGeometry(name);
		}
		delete[] _name;
	}
	qf.close();
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

}

void xdynamics_gui::xSave()
{

}

void xdynamics_gui::xOpen()
{
	QString file_path = QFileDialog::getOpenFileName(
		this, tr("open"), path,
		tr("View model file (*.vmd);;All files (*.*)"));
	int begin = file_path.lastIndexOf(".");
	QString ext = file_path.mid(begin + 1);
	if (ext == "vmd")
	{
		isOnViewModel = ReadViewModel(file_path);
	}
	else if (ext == "xls")
	{
		QString vf = ReadXLSFile(file_path);
		if (!vf.isEmpty())
			isOnViewModel = ReadViewModel(vf);
	}
}

QString xdynamics_gui::ReadXLSFile(QString xls_path)
{
	if (!xdm)
		xdm = new xDynamicsManager;
	xdm->OpenModelXLS(xls_path.toStdWString().c_str());
	int begin = xls_path.lastIndexOf("/");
	int end = xls_path.lastIndexOf(".");
	QString viewFile = path + xls_path.mid(begin + 1, end - begin-1) + ".vmd";
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
		int begin = s.lastIndexOf('.');
		QString ext = s.mid(begin+1);
		xcw->write(xCommandWindow::CMD_INFO, "File load - " + s);
		if (ext == "vmd")
			isOnViewModel = ReadViewModel(s);
		else if (ext == "rlt")
		{
			if (!isOnViewModel)
			{
				QString p = s.left(begin) + ".vmd";
				if (!ReadViewModel(p))
				{
					xcw->write(xCommandWindow::CMD_ERROR, kor("해석 결과에 부합하는 모델을 찾을 수 없습니다."));
					continue;
				}
				else isOnViewModel = true;
			}
			if (ReadModelResults(s)) setupAnimationTool();
		}
		else if (ext == "xls")
		{
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
}