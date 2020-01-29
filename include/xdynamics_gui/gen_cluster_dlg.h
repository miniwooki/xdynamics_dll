#pragma once
#include <q3dscatter.h>
#include <qscatterdataproxy.h>
#include <QDialog>
#include "ui_gen_cluster_dialog.h"

using namespace QtDataVisualization;

class ScatterDataModifier : public QObject
{
	Q_OBJECT
public:
	explicit ScatterDataModifier(Q3DScatter *scatter);
	~ScatterDataModifier();

	void setScale(double scale);
	void setRadius(int i, double radius, double scale);
	void removeLastParticle();
	void setPosition(int i, double x, double y, double z);
	void addParticle(int i, double x, double y, double z, double r, double scale);

public Q_SLOTS:
	//void setFieldLines(int lines);
	//void setArrowsPerLine(int arrows);
	//void toggleRotation();
	//void triggerRotation();
	//void toggleSun();

private:
	Q3DScatter *m_graph;
	QMap<int, QCustom3DItem*> m_particles;
	double maxRadius;
};

class gen_cluster_dlg : public QDialog, private Ui::GenClusterDialog
{
	Q_OBJECT
public:
	gen_cluster_dlg(QWidget* parent = nullptr);
	~gen_cluster_dlg();

private slots:
	void clickAdd();
	void clickApply();
	void clickCancel();
	void clickCell(int, int);
	void changeItem(QTableWidgetItem*);
	void increaseRows(int);
	void changeScale(double);

private:
	int rc[2];
	Q3DScatter* graph;
	ScatterDataModifier* modifier;
	QMap<QPair<int, int>, QTableWidgetItem*> tableItems;
};