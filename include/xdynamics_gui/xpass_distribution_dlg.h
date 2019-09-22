#pragma once
#include "ui_xpass_distribution.h"
#include <QStringList>

class xResultManager;

class xpass_distribution_dlg : public QDialog, private Ui::XPASS_DISTRIBUTION
{
	Q_OBJECT
public:
	xpass_distribution_dlg(QWidget* parent = NULL);
	~xpass_distribution_dlg();

	void setup(xResultManager* _xrm);
	QList<unsigned int>& get_distribution_result();

private slots:
	void click_select();
	void click_analysis();
	void click_exit();

private:
	xResultManager* xrm;
	QStringList qslist;
	QList<unsigned int> cid;
};