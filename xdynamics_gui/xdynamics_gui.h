#pragma once

#include <QtWidgets/QMainWindow>
#include <QtCore/QList>
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "ui_xdynamics_gui.h"
#include "xCommandWindow.h"
#include "xGLWidget.h"
#include "xAnimationTool.h"
#include "xModelNavigator.h"

class xdynamics_gui : public QMainWindow
{
	Q_OBJECT

public:
	enum { NEW = 0, OPEN, SAVE };
	
	xdynamics_gui(int _argc, char** _argv, QWidget *parent = Q_NULLPTR);
	~xdynamics_gui();

	bool ReadViewModel(QString path);
	bool ReadModelResults(QString path);

private slots:
	void xNew();
	void xSave();
	void xOpen();
	
private:
	void setupMainOperations();
	void setupAnimationTool();
	//void setupAnimationOperations();
	
	void dragEnterEvent(QDragEnterEvent *event);
	void dropEvent(QDropEvent *event);

	bool isOnViewModel;
	xAnimationTool* myAnimationBar;
	QList<QAction*> myMainActions;
	

	Ui::xdynamics_gui_mw ui;
	QString path;
	xGLWidget* xgl;
	xCommandWindow* xcw;
	xModelNavigator* xnavi;
};
