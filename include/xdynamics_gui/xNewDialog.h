#ifndef XNEWDIALOG_H
#define XNEWDIALOG_H

#include <QDialog>
#include "../xTypes.h"
#include "ui_xnew.h"

class xNewDialog : public QDialog, private Ui::DLG_NewModel
{
	Q_OBJECT

public:
	explicit xNewDialog(QWidget *parent = 0, QString cpath = QString());
	~xNewDialog();

	bool isBrowser;

	QString name;
	QString path;
	QString pathinbrowser;

	xUnitType unit;
	xGravityDirection dir_g;

	private slots:
	void Click_ok();
	void Click_browse();
};


#endif