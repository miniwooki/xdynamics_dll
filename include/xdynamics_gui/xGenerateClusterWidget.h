#pragma once

#include <Q3DScatter>
//#include "xdynamics_algebra/xAlgebraType.h"
#include "ui_wgenclusters.h"

using namespace QtDataVisualization;

class ScatterDataModifier;

class wgenclusters : public QWidget, public Ui::wgenclusterparticles
{
	Q_OBJECT

public:
	wgenclusters(QWidget* parent = NULL);
	~wgenclusters();

	//void set_starting_point(QString item, unsigned int sp);
	//bool get_enable_starting_point();
	//unsigned int get_starting_part();
	//QString get_starting_point_path();
	//bool is_simulationing();
	//void set_stop_state();
	//void set_start_state();
	void setClusterShapeName(QString name);
	QString ClusterShapeName();
	void setClusterView(unsigned int, double*);
	//void clickedStartPointButton();
	//void clickedStopButton();
signals:
	void clickedGenerateButton(QString, QString, int*, double*, bool);

private slots:
	void generateClusterParticles();
	void changeScale(int);
	//void StopButton();
	//void StartingPointButton();
	//void CheckStartingPoint(bool);

private:
	QString clusterShapeName;
	Q3DScatter *graph;
	ScatterDataModifier *modifier;
};
