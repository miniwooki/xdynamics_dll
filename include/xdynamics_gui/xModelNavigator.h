#ifndef XMODELNAVIGATOR_H
#define XMODELNAVIGATOR_H

#include <QDockWidget>
#include <QTreeWidget>
#include "ui_wcube.h"
#include "ui_wview.h"
#include "ui_wplane.h"
#include "ui_wparticles.h"
#include "xSimulationWidget.h"
#include "xPointMassWidget.h"
#include "xResultWidget.h"
//#include "ui_wsimulation.h"
//#include "xdynamics_gui.h"

class QScrollArea;
class xvObject;
//class xDynamicManager;

class wcube : public QWidget, public Ui::wcube
{
public:
	wcube(QWidget* parent = NULL) : QWidget(parent) { setupUi(this); }
	~wcube(){}

	//void setup(QString& name, xCubeObjectData* d);
};

class wplane : public QWidget, public Ui::wplane
{
public:
	wplane(QWidget* parent = NULL) : QWidget(parent) { setupUi(this); }
	~wplane(){}

	//void setup(QString& name, xCubeObjectData* d);
};

class wparticles : public QWidget, public Ui::wparticles
{
public:
	wparticles(QWidget* parent = NULL) : QWidget(parent) { setupUi(this); }
	~wparticles(){}

	//void setup(QString& name, xCubeObjectData* d);
};

class wview : public QWidget, public Ui::wview
{
	Q_OBJECT
public:
	wview(QWidget* parent = NULL) : QWidget(parent) 
	{ 
		setupUi(this);
		HSRed->setMaximum(255);
		HSGreen->setMaximum(255);
		HSBlue->setMaximum(255);
		LETransparency->setText("0");
		connect(HSTransparency, SIGNAL(valueChanged(int)), this, SLOT(changeTransparency(int)));
		connect(PBPalette, SIGNAL(clicked()), this, SLOT(colorPalette()));
		connect(HSRed, SIGNAL(valueChanged(int)), this, SLOT(changeRedColor(int)));
		connect(HSGreen, SIGNAL(valueChanged(int)), this, SLOT(changeGreenColor(int)));
		connect(HSBlue, SIGNAL(valueChanged(int)), this, SLOT(changeBlueColor(int)));
	}
	~wview(){}

	xvObject* xo;
	void setupColor();

	private slots :
	void changeTransparency(int);
		void colorPalette();
		void changeRedColor(int);
		void changeGreenColor(int);
		void changeBlueColor(int);
	
	//void setup(QString& name, xCubeObjectData* d);
};

class xModelNavigator : public QDockWidget
{
	Q_OBJECT

public:
	//enum tMotherRoot { OBJECT_ROOT = 0, RESULT_ROOT, SIMULATION_ROOT };
	enum tWidget{ NO_WIDGET = 0, CUBE_WIDGET, PLANE_WIDGET, SIMULATION_WIDGET, POINTMASS_WIDGET, PARTICLE_WIDGET, RESULT_WIDGET };
	enum tRoot { OBJECT_ROOT = 0, RESULT_ROOT, SIMULATION_ROOT, SHAPE_ROOT, MASS_ROOT, PARTICLE_ROOT, PART_ROOT };

	xModelNavigator();
	xModelNavigator(QWidget* parent);
	~xModelNavigator();
	static xModelNavigator* NAVIGATOR();
	void addChild(tRoot, QString _nm);
	void addChilds(tRoot, QStringList& qsl);
	QTreeWidgetItem* getRootItem(tRoot tr);
	void ClearTreeObject();
	void InitPlate();
	void deleteChild(tRoot, QString _nm);
	//static wsimulation* SimulationWidget();
signals:
	void definedSimulationWidget(wsimulation*);
	void definedPointMassWidget(wpointmass*);
	void definedResultWidget(wresult*);
	void InitializeWidgetStatement();

private slots:
//	void contextMenu(const QPoint&);
	void clickAction(QTreeWidgetItem*, int);

// signals:
// 	void propertySignal(QString, context_object_type);
private:
	void CallShape(QString& n);
	void CallParticles(QString& n);
	void CallPointMass(QString& n);
	//void CallView();
	void CallResultPart();
	void CallSimulation();
	void CallViewWidget(xvObject* xo);

	tWidget cwidget;
	QFrame *plate_frame;
	QVBoxLayout *plate_layout;
	QScrollArea *plate;
	QTreeWidget *vtree;
	//xDynamicsManager* xdm;
	QMap<tRoot, QTreeWidgetItem*> mom_roots;
	QMap<tRoot, QTreeWidgetItem*> roots;
	QMap<QString, int> idx_child;
};

//struct xCubeObjectData;


#endif