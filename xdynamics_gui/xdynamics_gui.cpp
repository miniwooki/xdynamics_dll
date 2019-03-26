#include "xdynamics_gui.h"
#include <QtCore/QDir>
#include <QtWidgets/QFileDialog>

xdynamics_gui::xdynamics_gui(int _argc, char** _argv, QWidget *parent)
	: QMainWindow(parent)
	, xgl(NULL)
	, xcw(NULL)
{
	path = QString::fromLocal8Bit(getenv("USERPROFILE"));
	path += +"/Documents/xdynamics/";
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
	setupMainOperations();
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

}

void xdynamics_gui::ReadViewModel(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
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
		}
		else if (vot == xViewObjectType::VPARTICLE)
		{
			xgl->vParticles()->defineFromViewFile(path);
			isExistParticle = true;
		}
	}
	qf.close();
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
		ReadViewModel(file_path);
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
