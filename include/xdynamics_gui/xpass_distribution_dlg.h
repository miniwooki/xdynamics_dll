#pragma once
#include "ui_xpass_distribution.h"
#include <QStringList>

class xResultManager;

typedef struct
{
	unsigned int id;
	float x, y, z;
	float cpx, cpy, cpz;
}distribution_data;

class xpass_distribution_dlg : public QDialog, private Ui::XPASS_DISTRIBUTION
{
	Q_OBJECT
public:
	xpass_distribution_dlg(QWidget* parent = NULL);
	~xpass_distribution_dlg();

	void setup(xResultManager* _xrm, QStringList qls);
	QMap<unsigned int, distribution_data>& get_distribution_result();

private slots:
	void click_select();
	void click_analysis();
	void click_exit();

private:
	xResultManager* xrm;
	QStringList frlist;
	QStringList qslist;
	QMap<unsigned int, distribution_data> cid;
};