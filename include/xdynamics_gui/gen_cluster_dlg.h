#pragma once
#include <q3dscatter.h>
#include <qscatterdataproxy.h>
#include <QDialog>
#include "ui_gen_cluster_dialog.h"

using namespace QtDataVisualization;

class xClusterObject;

class ScatterDataModifier : public QObject
{
	Q_OBJECT
public:
	explicit ScatterDataModifier(Q3DScatter *scatter);
	~ScatterDataModifier();

	void setScale(double scale);
	void setRadius(int i, double radius, double scale);
	void removeLastParticle();
	void setPosition(int i, double x, double y, double z, double scale);
	void addParticle(int i, double x, double y, double z, double r, double scale);
	void reset();
	unsigned int particleCount();

private slots:
	

private:
	Q3DScatter *m_graph;
	QMap<int, QCustom3DItem*> m_particles;
	QMap<int, double> m_radius;
	double maxRadius;
};



class gen_cluster_dlg : public QDialog, private Ui::GenClusterDialog
{
	typedef QMap<QPair<int, int>, QTableWidgetItem*> MapTableItems;
	Q_OBJECT
public:
	gen_cluster_dlg(QWidget* parent = nullptr);
	~gen_cluster_dlg();

	void prepareShow();
	QMap<QString, QPair<QString, xClusterObject*>> clusters;

private slots:
	void clickNew();
	void clickAdd();
	void clickGen();
	void clickCancel();
	void clickCell(int, int);
	void clickUpdate();
	void changeItem(QTableWidgetItem*);
	void changeCell(int r, int c);
	void increaseRows(int);
	void changeScale(int);
	void clickClusterItem(QListWidgetItem* item);

private:
	void setRowData(int i, QTableWidgetItem** items);
	void deleteCurrentTableItems();
	void loadExistLocalPosition(QString name);
	void checkNeedAdd();

private:
	//bool isExistInList;
	int rc[2];
	//bool isGenRandomOrientation;
	bool isClickedCell;
	bool isNewCluster;
	bool isChangeCluster;
	QListWidgetItem* current_list;
	Q3DScatter* graph;
	ScatterDataModifier* modifier;
	QMap<QPair<int, int>, QTableWidgetItem*> tableItems;
	QMap<QString, MapTableItems> tables;

};