#include "xNewDialog.h"
#include <QtWidgets>

xNewDialog::xNewDialog(QWidget* parent, QString cpath)
	: QDialog(parent)
	, isBrowser(false)
{
	setupUi(this);
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Browse, SIGNAL(clicked()), this, SLOT(Click_browse()));
	path = cpath;// model::path + "/";// kor(getenv("USERPROFILE"));
	//_path += "/Documents/xdynamics/";
	name = "Model1";
// 	if (!QDir(path).exists())
// 		QDir().mkdir(path);
	LE_Name->setText(name);
	//QFileDialog::getExistingDirectory(this, "", _path);
	CB_GravityDirection->setCurrentIndex(4);
}

xNewDialog::~xNewDialog()
{
	//disconnect(PB_Ok);
	//disconnect(PBBrowse);
	//if (ui) delete ui; ui = NULL;
}

void xNewDialog::Click_ok()
{
	name = LE_Name->text();
	//_path = ui->LEPath->text();
	unit = (xUnitType)CB_Unit->currentIndex();
	dir_g = (xGravityDirection)CB_GravityDirection->currentIndex();
	//_path += "/";
	this->close();
	this->setResult(QDialog::Accepted);
}

void xNewDialog::Click_browse()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("open"), path, tr("Ecxel File(*.xls)"));
	if (!fileName.isEmpty()){
// 		pathinbrowser = fileName;
// 		int begin = fileName.lastIndexOf("/");
// 		int end = fileName.lastIndexOf(".");
	//	name = fileName.mid(begin + 1, end - begin - 1);
		//path = fileName.mid(0, begin + 1);
	}
	else
		return;
	pathinbrowser = fileName;
	isBrowser = true;
	this->close();
	this->setResult(QDialog::Accepted);
}