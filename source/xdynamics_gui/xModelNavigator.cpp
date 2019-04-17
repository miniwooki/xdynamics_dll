#include "xModelNavigator.h"
#include "xGLWidget.h"
//#include "modelManager.h"
//#include "glwidget.h"
#include "xvCube.h"
#include "xvPlane.h"
#include "xdynamics_simulation/xSimulation.h"
//#include "ui_wcube.h"
#include <QMenu>
#include <QFrame>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QDebug>
#include <QScrollArea>
#include <QColorDialog>

xModelNavigator* db;
// wcube* wc = NULL;
// wplane* wp = NULL;
// wsimulation* xws = NULL;
wsimulation::wsimulation(QWidget* parent /* = NULL */)
	: QWidget(parent)
{

}

xModelNavigator::xModelNavigator()
	: plate(NULL)
	, vtree(NULL)
	, wv(NULL)
	, plate_frame(NULL)
	, plate_layout(NULL)
	//, xdm(NULL)
	, cwidget(NO_WIDGET)
{

}

xModelNavigator::xModelNavigator(QWidget* parent)
	: QDockWidget(parent)
	, plate(NULL)
	, vtree(NULL)
	, wv(NULL)
	, plate_frame(NULL)
	, plate_layout(NULL)
	//, xdm(NULL)
	, cwidget(NO_WIDGET)
{
	db = this;
	QFrame *frame = new QFrame(this);
	QVBoxLayout *layout = new QVBoxLayout;
	vtree = new QTreeWidget;
	plate = new QScrollArea;
	plate_frame = new QFrame;
	plate_layout = new QVBoxLayout;
	layout->addWidget(vtree);
	layout->addWidget(plate);
	vtree->setColumnCount(1);
	vtree->setHeaderLabel("Navigator");
	vtree->setContextMenuPolicy(Qt::CustomContextMenu);
	
	//vtree->setWindowTitle("xModelNavigator");
	//setWidget();
	mom_roots[OBJECT_ROOT] = new QTreeWidgetItem(vtree); mom_roots[OBJECT_ROOT]->setText(0, "Objects");
	roots[SHAPE_ROOT] = new QTreeWidgetItem(mom_roots[OBJECT_ROOT]); roots[SHAPE_ROOT]->setText(0, "Shape");// addChild(OBJECT_ROOT, "Shape");
	roots[MASS_ROOT] = new QTreeWidgetItem(mom_roots[OBJECT_ROOT]); roots[MASS_ROOT]->setText(0, "Mass");// addChild(OBJECT_ROOT, "Mass");
	roots[PARTICLE_ROOT] = new QTreeWidgetItem(mom_roots[OBJECT_ROOT]); roots[PARTICLE_ROOT]->setText(0, "Particle"); //addChild(OBJECT_ROOT, "Particle");
	//roots[PART_ROOT] = new QTreeWidgetItem(); addChild(OBJECT_ROOT)
	mom_roots[OBJECT_ROOT]->setExpanded(true);
	mom_roots[RESULT_ROOT] = new QTreeWidgetItem(vtree); mom_roots[RESULT_ROOT]->setText(0, "Results");
	mom_roots[SIMULATION_ROOT] = new QTreeWidgetItem(vtree); mom_roots[SIMULATION_ROOT]->setText(0, "Simulation");
	

// 	roots[SHAPE_ROOT]->setText(0, "Shape");
// 	roots[MASS_ROOT]->setText(0, "Mass");
// 	roots[PARTICLE_ROOT]->setText(0, "Particle");
// 	roots[PART_ROOT]->setText(0, "Part");
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
// 	plate_frame->setMaximumWidth(220);
// 	plate_frame->setLayout(plate_layout);
// 	//QGridLayout *glayout = new QGridLayout;
// 	//glayout->addWidget(frame);
// 	//plate->setLayout(glayout);
// 	plate->setWidget(plate_frame);
	//xws = new wsimulation;
	//wv = new wview;
	
}

xModelNavigator::~xModelNavigator()
{
	//qDeleteAll(mom_roots.begin(), mom_roots.end());
	qDeleteAll(roots);
	if (vtree) delete vtree; vtree = NULL;
	
	//RemovePlateWidget();
	if (plate_layout) delete plate_layout; plate_layout = NULL;
	if (plate_frame) delete plate_frame; plate_frame = NULL;
	if (plate) delete plate; plate = NULL;
	//if (wv) delete wv; wv = NULL;
//	if (xws) delete xws; xws = NULL;
	//if (wp) delete wp; wp = NULL;
	

	//if (plate_layout) delete plate_layout; plate_layout = NULL;
}

// xModelNavigator* xModelNavigator::DB()
// {
// 	return db;
// }

void xModelNavigator::addChild(tRoot tr, QString _nm)
{
	QTreeWidgetItem* parent = getRootItem(tr);
	if (!parent) return;
	///QString t = parent->text(1);
	QTreeWidgetItem* child = new QTreeWidgetItem();
	child->setText(0, _nm);
	//child->setData(0, (int)tr, v);
	parent->addChild(child);		
	parent->setExpanded(true);
}

void xModelNavigator::addChilds(tRoot tr, QStringList& qsl)
{
	foreach(QString s, qsl)
	{
		addChild(tr, s);
	}
}

QTreeWidgetItem* xModelNavigator::getRootItem(tRoot tr)
{
	QList<tRoot> keys = mom_roots.keys();
	QList<tRoot>::const_iterator it = qFind(keys, tr);
	if (it == keys.end() || !keys.size())
	{
		keys = roots.keys();
		it = qFind(keys, tr);
		if (it == keys.end() || !keys.size())
			return NULL;
		return roots[tr];
	}
	return mom_roots[tr];
}

// wsimulation* xModelNavigator::SimulationWidget()
// {
// 	return xws;
// }

// void xModelNavigator::setDynamicManager(xDynamicsManager* _xdm)
// {
// 	xdm = _xdm;
// }

void xModelNavigator::clickAction(QTreeWidgetItem* w, int i)
{
	//tRoot tr = w->data(i, (int)w)
	if (plate_layout) delete plate_layout;
	if (plate_frame) delete plate_frame;
	
	plate_frame = new QFrame;
	plate_layout = new QVBoxLayout;
	QTreeWidgetItem *parent = w->parent();
	//RemovePlateWidget();
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
		}
	}
	else
	{
		tRoot tr = mom_roots.key(w);
		QString name = w->text(i);
		int type = w->data(i, (int)tr).toInt();
		qDebug() << "Clicked item : " << tr << " - " << name;
		switch (tr)
		{
		case PART_ROOT: break;
		case SIMULATION_ROOT: CallSimulation(); break;
		}
	}
}

void xModelNavigator::CallShape(QString& n)
{
	xvObject* xo = xGLWidget::GLObject()->Object(n);
	//QFrame* frame = new QFrame(plate);
	if (xo->ObjectType() == xvObject::V_CUBE)
	{
		xCubeObjectData d = { 0, };
		d = dynamic_cast<xvCube*>(xo)->CubeData();
		////if (!wc)
		wcube *wc = new wcube(plate);
		wc->LEName->setText(n);
		wc->LEP1X->setText(QString("%1").arg(d.p0x)); wc->LEP1Y->setText(QString("%1").arg(d.p0y)); wc->LEP1Z->setText(QString("%1").arg(d.p0z));
		wc->LEP2X->setText(QString("%1").arg(d.p1x)); wc->LEP2Y->setText(QString("%1").arg(d.p1y)); wc->LEP2Z->setText(QString("%1").arg(d.p1z));
		wc->LESZX->setText(QString("%1").arg(d.p1x - d.p0x));
		wc->LESZY->setText(QString("%1").arg(d.p1y - d.p0y));
		wc->LESZZ->setText(QString("%1").arg(d.p1z - d.p0z));
		plate_layout->addWidget(wc);
		cwidget = CUBE_WIDGET;
	}
	else if (xo->ObjectType() == xvObject::V_PLANE)
	{
		xPlaneObjectData d = { 0, };
		d = dynamic_cast<xvPlane*>(xo)->PlaneData();
// 		if (wp)
// 			delete wp;
		wplane *wp = new wplane(plate);
		wp->LEName->setText(n);
		wp->LEP1X->setText(QString("%1").arg(d.pox)); wp->LEP1Y->setText(QString("%1").arg(d.poy)); wp->LEP1Z->setText(QString("%1").arg(d.poz));
		wp->LEP2X->setText(QString("%1").arg(d.p1x)); wp->LEP2Y->setText(QString("%1").arg(d.p1y)); wp->LEP2Z->setText(QString("%1").arg(d.p1z));
		wp->LEDIRX->setText(QString("%1").arg(d.drx)); wp->LEDIRY->setText(QString("%1").arg(d.dry)); wp->LEDIRZ->setText(QString("%1").arg(d.drz));
		plate_layout->addWidget(wp);
		cwidget = PLANE_WIDGET;
	}
	
	CallViewWidget(xo);
	
}


void xModelNavigator::CallSimulation()
{
// 	if (!xdm)
// 		return false;
	double dt = xSimulation::dt;
	unsigned int st = xSimulation::st;
	double et = xSimulation::et;
	//if (!xws)
	wsimulation *xws = new wsimulation(/*plate*/);
	xws->LETimeStep->setText(QString("%1").arg(dt));
	xws->LESaveStep->setText(QString("%1").arg(st));
	xws->LEEndTime->setText(QString("%1").arg(et));
	//xws->setParent(plate);
	plate_layout->addWidget(xws);
	plate_layout->setAlignment(Qt::AlignTop);
	plate_frame->setMaximumWidth(240);
	plate_frame->setLayout(plate_layout);
	plate->setWidget(plate_frame);
	cwidget = SIMULATION_WIDGET;
}

void xModelNavigator::CallViewWidget(xvObject* xo)
{
// 	if (wv)
// 		delete wv;
	wv = new wview(plate);
	wv->xo = xo;
	plate_layout->addWidget(wv);
	plate_layout->setAlignment(Qt::AlignTop);
	plate_layout->setMargin(0);
	plate_layout->setStretch(0, 0);
	plate_frame->setMaximumWidth(240);
	plate_frame->setLayout(plate_layout);
	//QGridLayout *glayout = new QGridLayout;
	//glayout->addWidget(frame);
	//plate->setLayout(glayout);
	plate->setWidget(plate_frame);
	wv->setupColor();
}

void xModelNavigator::RemovePlateWidget()
{
// 	switch (cwidget)
// 	{
// 	case CUBE_WIDGET:
// 		//plate_layout->wi
// 		plate_layout->removeWidget(wc);
// 		plate_layout->removeWidget(wv);
// 		break;
// 	case SIMULATION_WIDGET:
// 		plate_layout->removeWidget(xws);
// 		break;
// 	case PLANE_WIDGET:
// 		plate_layout->removeWidget(wp);
// 		plate_layout->removeWidget(wv);
// 		break;
// 	}
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
