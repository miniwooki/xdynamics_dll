#pragma once

#include <QtWidgets/QMainWindow>
#include <QtCore/QList>
#include "ui_xdynamics_gui.h"
#include "xCommandWindow.h"
#include "xGLWidget.h"

class xdynamics_gui : public QMainWindow
{
	Q_OBJECT

public:
	enum { NEW = 0, OPEN, SAVE };
	xdynamics_gui(int _argc, char** _argv, QWidget *parent = Q_NULLPTR);
	~xdynamics_gui();

	void ReadViewModel(QString path);

private slots:
	void xNew();
	void xSave();
	void xOpen();

private:
	void setupMainOperations();

	QList<QAction*> myMainActions;

	Ui::xdynamics_gui_mw ui;
	QString path;
	xGLWidget* xgl;
	xCommandWindow* xcw;
};
