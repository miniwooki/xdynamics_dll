#ifndef XMODELNAVIGATOR_H
#define XMODELNAVIGATOR_H

#include <QDockWidget>
#include <QTreeWidget>
#include "ui_wcube.h"
#include "ui_wview.h"

class QScrollArea;
class xvObject;

class wcube : public QWidget, public Ui::wcube
{
public:
	wcube(QWidget* parent = NULL) : QWidget(parent) { setupUi(this); }
	~wcube(){}

	//void setup(QString& name, xCubeObjectData* d);
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
	enum tRoot { OBJECT_ROOT = 0, RESULT_ROOT, SIMULATION_ROOT, SHAPE_ROOT, MASS_ROOT, PARTICLE_ROOT, PART_ROOT };

	xModelNavigator();
	xModelNavigator(QWidget* parent);
	~xModelNavigator();

	void addChild(tRoot, QString _nm);
	void addChilds(tRoot, QStringList& qsl);
	QTreeWidgetItem* getRootItem(tRoot tr);

private slots:
//	void contextMenu(const QPoint&);
	void clickAction(QTreeWidgetItem*, int);

// signals:
// 	void propertySignal(QString, context_object_type);
private:
	void CallShape(QString& n);
	//void CallView();
	void CallViewWidget();

	
	wview *wv;
	QFrame *plate_frame;
	QVBoxLayout *plate_layout;
	QScrollArea *plate;
	QTreeWidget *vtree;
	QMap<tRoot, QTreeWidgetItem*> mom_roots;
	QMap<tRoot, QTreeWidgetItem*> roots;
};

//struct xCubeObjectData;


#endif