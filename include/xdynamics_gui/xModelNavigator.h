#ifndef XMODELNAVIGATOR_H
#define XMODELNAVIGATOR_H

#include <QDockWidget>
#include <QTreeWidget>
#include "ui_wcube.h"
#include "ui_wview.h"
#include "ui_wplane.h"
#include "ui_wsimulation.h"

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

class wsimulation : public QWidget, public Ui::wsimulation
{
	Q_OBJECT
public:
	wsimulation(QWidget* parent = NULL);// { setupUi(this); }
	~wsimulation(){}

	signals:
	void clickedSolveButton();

	private slots:
	void SolveButton() { emit clickedSolveButton(); }
};

class wview : public QWidget, public Ui::wview
{
	Q_OBJECT
public:
	wview(QWidget* parent = NULL) : QWidget(parent) 
	{ 
		setupUi(this);
		connect(PBPalette, SIGNAL(clicked()), this, SLOT(colorPalette()));
		connect(HSRed, SIGNAL(valueChanged(int)), this, SLOT(changeRedColor(int)));
		connect(HSGreen, SIGNAL(valueChanged(int)), this, SLOT(changeGreenColor(int)));
		connect(HSBlue, SIGNAL(valueChanged(int)), this, SLOT(changeBlueColor(int)));
	}
	~wview(){}

	xvObject* xo;
	void setupColor();

	private slots :
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
	enum tWidget{ NO_WIDGET = 0, CUBE_WIDGET, PLANE_WIDGET, SIMULATION_WIDGET };
	enum tRoot { OBJECT_ROOT = 0, RESULT_ROOT, SIMULATION_ROOT, SHAPE_ROOT, MASS_ROOT, PARTICLE_ROOT, PART_ROOT };

	xModelNavigator();
	xModelNavigator(QWidget* parent);
	~xModelNavigator();

	void addChild(tRoot, QString _nm);
	void addChilds(tRoot, QStringList& qsl);
	QTreeWidgetItem* getRootItem(tRoot tr);
	//void setDynamicManager(xDynamicsManager* _xdm);
	static wsimulation* SimulationWidget();
private slots:
//	void contextMenu(const QPoint&);
	void clickAction(QTreeWidgetItem*, int);

// signals:
// 	void propertySignal(QString, context_object_type);
private:
	void CallShape(QString& n);
	//void CallView();
	void CallSimulation();
	void CallViewWidget(xvObject* xo);
	void RemovePlateWidget();

	tWidget cwidget;
	wview *wv;
	QFrame *plate_frame;
	QVBoxLayout *plate_layout;
	QScrollArea *plate;
	QTreeWidget *vtree;
	//xDynamicsManager* xdm;
	QMap<tRoot, QTreeWidgetItem*> mom_roots;
	QMap<tRoot, QTreeWidgetItem*> roots;
};

//struct xCubeObjectData;


#endif