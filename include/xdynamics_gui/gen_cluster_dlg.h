#pragma once
#include <QtDataVisualization/q3dscatter.h>
#include <QtDataVisualization/qscatterdataproxy.h>
#include <QDialog>
#include "ui_gen_cluster_dialog.h"

using namespace QtDataVisualization;

class ScatterDataModifier : public QObject
{
	Q_OBJECT
public:
	explicit ScatterDataModifier(Q3DScatter *scatter);
	~ScatterDataModifier();

	void addParticle(int i, double x, double y, double z, double r);

public Q_SLOTS:
	//void setFieldLines(int lines);
	//void setArrowsPerLine(int arrows);
	//void toggleRotation();
	//void triggerRotation();
	//void toggleSun();

private:
	Q3DScatter *m_graph;
	QMap<int, QCustom3DItem*> m_particles;
	QVector3D minAxis;
	QVector3D maxAxis;
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
	void increaseRows(int);

private:
	Q3DScatter* graph;
	ScatterDataModifier* modifier;
	QMap<QPair<int, int>, QTableWidgetItem*> tableItems;
};