#include "xModelNavigator.h"
#include "xGLWidget.h"
//#include "modelManager.h"
//#include "glwidget.h"
#include "xvCube.h"
#include "ui_wcube.h"
#include <QMenu>
#include <QFrame>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QDebug>
#include <QScrollArea>
#include <QColorDialog>

xModelNavigator* db;

xModelNavigator::xModelNavigator()
	: plate(NULL)
	, vtree(NULL)
	, wv(NULL)
	, plate_frame(NULL)
	, plate_layout(NULL)
{

}

xModelNavigator::xModelNavigator(QWidget* parent)
	: QDockWidget(parent)
	, plate(NULL)
	, vtree(NULL)
	, wv(NULL)
	, plate_frame(NULL)
	, plate_layout(NULL)
{
	db = this;
	QFrame *frame = new QFrame(this);
	QVBoxLayout *layout = new QVBoxLayout;
	vtree = new QTreeWidget;
	plate = new QScrollArea;
	layout->addWidget(vtree);
	layout->addWidget(plate);
	vtree->setColumnCount(1);
	vtree->setHeaderLabel("Navigator");
	vtree->setContextMenuPolicy(Qt::CustomContextMenu);
	//vtree->setWindowTitle("xModelNavigator");
	//setWidget();
	roots[SHAPE_ROOT] = new QTreeWidgetItem(vtree);
	roots[MASS_ROOT] = new QTreeWidgetItem(vtree);
	roots[PARTICLE_ROOT] = new QTreeWidgetItem(vtree);
	roots[PART_ROOT] = new QTreeWidgetItem(vtree);

	roots[SHAPE_ROOT]->setText(0, "Shape");
	roots[MASS_ROOT]->setText(0, "Mass");
	roots[PARTICLE_ROOT]->setText(0, "Particle");
	roots[PART_ROOT]->setText(0, "Part");
	layout->setMargin(0);
	frame->setLayout(layout);
	this->setWidget(frame);
	this->setMinimumWidth(240);
	this->setMaximumWidth(240);
	//plate->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	
	connect(vtree, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(clickAction(QTreeWidgetItem*, int)));
// 	roots[PLANE_ROOT]->setIcon(0, QIcon(":/Resources/icRect.png"));
// 	roots[LINE_ROOT]->setIcon(0, QIcon(":/Resources/icLine.png"));
// 	roots[CUBE_ROOT]->setIcon(0, QIcon(":/Resources/pRec.png"));
// 	roots[CYLINDER_ROOT]->setIcon(0, QIcon(":/Resources/cylinder.png"));
// 	roots[POLYGON_ROOT]->setIcon(0, QIcon(":/Resources/icPolygon.png"));
// 	roots[RIGID_BODY_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));
// 	roots[COLLISION_ROOT]->setIcon(0, QIcon(":/Resources/collision.png"));
// 	roots[PARTICLES_ROOT]->setIcon(0, QIcon(":/Resources/particle.png"));
// 	roots[CONSTRAINT_ROOT]->setIcon(0, QIcon(":/Resources/spherical.png"));
// 	roots[SPRING_DAMPER_ROOT]->setIcon(0, QIcon(":/Resources/TSDA_icon.png"));
	//connect(vtree, &QTreeWidget::customContextMenuRequested, this, &xModelNavigator::contextMenu);
//	md->setxModelNavigator(this);
}

xModelNavigator::~xModelNavigator()
{
	qDeleteAll(roots.begin(), roots.end());
	if (vtree) delete vtree; vtree = NULL;
	if (wv) delete wv; wv = NULL;
	if (plate_frame) delete plate_frame; plate_frame = NULL;
	//if (plate_layout) delete plate_layout; plate_layout = NULL;
}

// xModelNavigator* xModelNavigator::DB()
// {
// 	return db;
// }

void xModelNavigator::addChild(tRoot tr, QString& _nm)
{
	QTreeWidgetItem* child = new QTreeWidgetItem();
	child->setText(0, _nm);
	//child->setData(0, (int)tr, v);
	roots[tr]->addChild(child);		
}

void xModelNavigator::addChilds(tRoot tr, QStringList& qsl)
{
	foreach(QString s, qsl)
	{
		addChild(tr, s);
	}
}

void xModelNavigator::clickAction(QTreeWidgetItem* w, int i)
{
	//tRoot tr = w->data(i, (int)w)
	QTreeWidgetItem *parent = w->parent();
	if (parent)
	{
		tRoot tr = roots.key(parent);
		QString name = w->text(i);
		int type = w->data(i, (int)tr).toInt();
		qDebug() << "Clicked item : " << tr << " - " << name;
		switch (tr)
		{
		case SHAPE_ROOT: CallShape(name); break;
		case MASS_ROOT: break;
		case PARTICLE_ROOT: break;
		case PART_ROOT: break;
		}
	}
}

void xModelNavigator::CallShape(QString& n)
{
	xvObject* xo = xGLWidget::GLObject()->Object(n);
	//QFrame* frame = new QFrame(plate);
	plate_layout = new QVBoxLayout;
	if (xo->ObjectType() == xvObject::V_CUBE)
	{
		xCubeObjectData d = { 0, };
		d = dynamic_cast<xvCube*>(xo)->CubeData();
		wcube *wc = new wcube(plate);
		wc->LEName->setText(n);
		wc->LEP1X->setText(QString("%1").arg(d.p0x)); wc->LEP1Y->setText(QString("%1").arg(d.p0y)); wc->LEP1Z->setText(QString("%1").arg(d.p0z));
		wc->LEP2X->setText(QString("%1").arg(d.p1x)); wc->LEP2Y->setText(QString("%1").arg(d.p1y)); wc->LEP2Z->setText(QString("%1").arg(d.p1z));
		wc->LESZX->setText(QString("%1").arg(d.p1x - d.p0x));
		wc->LESZY->setText(QString("%1").arg(d.p1y - d.p0y));
		wc->LESZZ->setText(QString("%1").arg(d.p1z - d.p0z));
		plate_layout->addWidget(wc);
	}
	wv->xo = xo;
	CallViewWidget();
	
	//frame->setLayout()
	//plate->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	//plate->setAlignment(Qt::AlignTop);
}


void xModelNavigator::CallViewWidget()
{
	if (!wv)
	{
		wv = new wview(plate);
		plate_layout->addWidget(wv);
		plate_layout->setAlignment(Qt::AlignTop);
		plate_frame = new QFrame;
		plate_frame->setMaximumWidth(220);
		plate_frame->setLayout(plate_layout);
		//QGridLayout *glayout = new QGridLayout;
		//glayout->addWidget(frame);
		//plate->setLayout(glayout);
		plate->setWidget(plate_frame);
	}
	wv->setupColor();
}

// void xModelNavigator::CallView()
// {
// 
// }

// void xModelNavigator::contextMenu(const QPoint& pos)
// {
// 	QTreeWidgetItem* item = vtree->itemAt(pos);
// 	int col = vtree->currentColumn();
// 	if (!item)
// 		return;
// 	if (!item->parent())
// 		return;
// 	QString it = item->text(0);
// 	QMenu menu(item->text(0), this);
// 	menu.addAction("Delete");
// 	menu.addAction("Property");
// 
// 	QPoint pt(pos);
// 	QAction *a = menu.exec(vtree->mapToGlobal(pos));
// 
// 	if (a)
// 	{
// 		QString txt = a->text();
// 		if (txt == "Delete")
// 		{
// 			QTreeWidgetItem* parent = item->parent();
// 			modelManager::MM()->ActionDelete(item->text(0));
// 			GLWidget::GLObject()->actionDelete(item->text(0));
// 			parent->removeChild(item);
// 			delete item;
// 		}
// 		else if (txt == "Property")
// 		{
// 
// 		}
// 	}
// 	menu.clear();
	//	qDeleteAll(menu);
//}

// void xModelNavigator::actProperty()
// {
// 	//return;
// }
// 
// void xModelNavigator::actDelete()
// {
// 	
// }


// wcube

//  void wcube::setup(QString& name, xCubeObjectData* d)
//{
//	LEName->setText(name);
//}

// wview
void wview::setupColor()
{
	QColor c = xo->Color();
	HSRed->setValue(c.red());
	HSGreen->setValue(c.green());
	HSBlue->setValue(c.blue());
	LERed->setText(QString("%1").arg(c.red()));
	LEGreen->setText(QString("%1").arg(c.green()));
	LEBlue->setText(QString("%1").arg(c.blue()));
}

void wview::colorPalette()
{
	QColor c = QColorDialog::getColor();
	int red = c.red();
	int green = c.green();
	int blue = c.blue();
	HSRed->setValue(red);
	HSGreen->setValue(green);
	HSBlue->setValue(blue);
	LERed->setText(QString("%1").arg(red));
	LEGreen->setText(QString("%1").arg(green));
	LEBlue->setText(QString("%1").arg(blue));
	
}

void wview::changeRedColor(int r)
{
	LERed->setText(QString("%1").arg(r));
}

void wview::changeGreenColor(int g)
{
	LEGreen->setText(QString("%1").arg(g));
}

void wview::changeBlueColor(int b)
{
	LEBlue->setText(QString("%1").arg(b));
}
