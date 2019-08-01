//#include "xdynamics_gui.h"
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

xModelNavigator::xModelNavigator()
	: plate(NULL)
	, vtree(NULL)
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
	this->setMinimumWidth(270);
	this->setMaximumWidth(270);
	//plate->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	
	connect(vtree, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(clickAction(QTreeWidgetItem*, int)));
}

xModelNavigator::~xModelNavigator()
{
	qDeleteAll(roots);
	if (vtree) delete vtree; vtree = NULL;
	if (plate_layout) delete plate_layout; plate_layout = NULL;
	if (plate_frame) delete plate_frame; plate_frame = NULL;
	if (plate) delete plate; plate = NULL;
}

xModelNavigator* xModelNavigator::NAVIGATOR()
{
	return db;
}

void xModelNavigator::ClearTreeObject()
{
	QList<QTreeWidgetItem*> result_root, shape_root, mass_root, particle_root;
	result_root = mom_roots[RESULT_ROOT]->takeChildren(); 
	if (result_root.size()) qDeleteAll(result_root);
	shape_root = roots[SHAPE_ROOT]->takeChildren();
	if (shape_root.size()) qDeleteAll(shape_root);
	mass_root = roots[MASS_ROOT]->takeChildren();
	if (mass_root.size()) qDeleteAll(mass_root);
	particle_root = roots[PARTICLE_ROOT]->takeChildren();
	if (particle_root.size()) qDeleteAll(particle_root);
	if (plate_layout) delete plate_layout; plate_layout = NULL;
	if (plate_frame) delete plate_frame; plate_frame = NULL;
}

void xModelNavigator::InitPlate()
{

}

void xModelNavigator::deleteChild(tRoot t, QString _nm)
{
	QTreeWidgetItem* it = roots[t]->takeChild(idx_child[_nm]);
	if (it)
		delete it;
	idx_child.take(_nm);
}

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
	idx_child[_nm] = parent->indexOfChild(child);
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
	emit InitializeWidgetStatement();
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
		case MASS_ROOT: CallPointMass(name); break;
		case RESULT_ROOT: CallResultPart(); break;
		case PARTICLE_ROOT: CallParticles(name); break;
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
		wc->setMinimumWidth(230);
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
		wp->setMinimumWidth(250);
		vector3d p0 = new_vector3d(d.p0x, d.p0y, d.p0z);
		vector3d p1 = new_vector3d(d.p1x, d.p1y, d.p1z);
		vector3d p3 = new_vector3d(d.p3x, d.p3y, d.p3z);
		vector3d dir = cross(p1 - p0, p3 - p0);
		dir = dir / length(dir);
		wp->LEP1X->setText(QString("%1").arg(d.p0x)); wp->LEP1Y->setText(QString("%1").arg(d.p0y)); wp->LEP1Z->setText(QString("%1").arg(d.p0z));
		wp->LEP2X->setText(QString("%1").arg(d.p2x)); wp->LEP2Y->setText(QString("%1").arg(d.p2y)); wp->LEP2Z->setText(QString("%1").arg(d.p2z));
		wp->LEDIRX->setText(QString("%1").arg(dir.x)); wp->LEDIRY->setText(QString("%1").arg(dir.y)); wp->LEDIRZ->setText(QString("%1").arg(dir.z));
		plate_layout->addWidget(wp);
		cwidget = PLANE_WIDGET;
	}
	CallViewWidget(xo);
}


void xModelNavigator::CallParticles(QString& n)
{
	xvParticle* xp = xGLWidget::GLObject()->vParticles();
	if (xp)
	{
		QString nm = xp->NameOfGroupData(n);
		QString mat = xp->MaterialOfGroupData(n);
		unsigned int num_this = xp->NumParticlesOfGroupData(n);
		unsigned int num_total = xp->NumParticles();
		double min_rad = xp->MinRadiusOfGroupData(n);
		double max_rad = xp->MaxnRadiusOfGroupData(n);
		wparticles *wp = new wparticles(plate);
		wp->LEName->setText(n);
		wp->LEMaterial->setText(mat);
		wp->LENumThis->setText(QString("%1").arg(num_this));
		wp->LENumTotal->setText(QString("%1").arg(num_total));
		wp->LEMinRadius->setText(QString("%1").arg(min_rad));
		wp->LEMaxRadius->setText(QString("%1").arg(max_rad));
		plate_layout->addWidget(wp);
		plate_layout->setAlignment(Qt::AlignTop);
		plate_frame->setMaximumWidth(270);
		plate_frame->setLayout(plate_layout);
	//	plate_layout->addWidget(wp);
		plate->setWidget(plate_frame);
		cwidget = PARTICLE_WIDGET;
	}
	//CallViewWidget(n)
}

void xModelNavigator::CallResultPart()
{
	wresult *xr = new wresult(plate);
	xr->LE_LimitMin->setText("0.0");
	xr->LE_LimitMax->setText("0.0");
	xr->RB_UserInput->setChecked(false);
	xr->RB_FromResult->setChecked(true);
	//xws->LEEndTime->setText("0.0");
	//xws->UpdateInformation();
	////xws->setParent(plate);
	plate_layout->addWidget(xr);
	plate_layout->setAlignment(Qt::AlignTop);
	plate_frame->setMaximumWidth(270);
	plate_frame->setLayout(plate_layout);
	plate->setWidget(plate_frame);
	cwidget = RESULT_WIDGET;
}

void xModelNavigator::CallSimulation()
{
// 	if (!xdm)
// 		return false;
	double dt = xSimulation::dt;
	unsigned int st = xSimulation::st;
	double et = xSimulation::et;
	//if (!xws)
	wsimulation *xws = new wsimulation(plate);
	xws->LETimeStep->setText(QString("%1").arg(dt));
	xws->LESaveStep->setText(QString("%1").arg(st));
	xws->LEEndTime->setText(QString("%1").arg(et));
	xws->UpdateInformation();
	//xws->setParent(plate);
	plate_layout->addWidget(xws);
	plate_layout->setAlignment(Qt::AlignTop);
	plate_frame->setMaximumWidth(270);
	plate_frame->setLayout(plate_layout);
	plate->setWidget(plate_frame);
	cwidget = SIMULATION_WIDGET;

	emit definedSimulationWidget(xws);
}

void xModelNavigator::CallPointMass(QString& n)
{
	wpointmass* xpm = new wpointmass(plate);
	xpm->LEName->setText(n);
	plate_layout->addWidget(xpm);
	plate_layout->setAlignment(Qt::AlignTop);
	plate_frame->setMaximumWidth(270);
	plate_frame->setLayout(plate_layout);
	plate->setWidget(plate_frame);
	cwidget = POINTMASS_WIDGET;

	emit definedPointMassWidget(xpm);
}

void xModelNavigator::CallViewWidget(xvObject* xo)
{
// 	if (wv)
// 		delete wv;
	wview *wv = new wview(plate);
	wv->setMinimumWidth(250);
	wv->xo = xo;
	plate_layout->addWidget(wv);
	plate_layout->setAlignment(Qt::AlignTop);
	//plate_layout->setMargin(0);
	plate_frame->setMaximumWidth(270);
	plate_frame->setLayout(plate_layout);
	plate->setWidget(plate_frame);
	wv->setupColor();
}

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

void wview::changeTransparency(int c)
{
	xo->setBlendAlpha((100 - c) * 0.01);
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
	xo->setColor(c);
}

void wview::changeRedColor(int r)
{
	LERed->setText(QString("%1").arg(r));
	xo->Color().setRed(r);
}

void wview::changeGreenColor(int g)
{
	LEGreen->setText(QString("%1").arg(g));
	xo->Color().setGreen(g);
}

void wview::changeBlueColor(int b)
{
	LEBlue->setText(QString("%1").arg(b));
	xo->Color().setBlue(b);
}
