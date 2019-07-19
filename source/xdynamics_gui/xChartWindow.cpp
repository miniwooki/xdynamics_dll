#include "xChartWindow.h"
//#include "model.h"
//#include "waveHeightSensor.h"
#include "xCallOut.h"
//#include "cmdWindow.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include <QtCharts/QLineSeries>
#include <QtWidgets>
#include <QtCharts>
#include <QtGui>
#include <QToolBar>
#include <QMap>

bool xChartWindow::isActivate = false;

xChartWindow::xChartWindow(QWidget* parent /* = NULL */)
	: QMainWindow(parent)
	, commandStatus(0)
	, xSize(700)
	, ySize(600)
	, openColumnCount(1)
	, wWidth(0)
	, wHeight(0)
	, isAutoUpdateProperties(true)
	, isEditingCommand(false)
	, vcht(NULL)
	//	, rs(NULL)
	, cmodel(NULL)
	, m_tooltip(NULL)
	, mainToolBar(NULL)
	//, prop(NULL)
	, comm(NULL)
	//, whs(NULL)
	, commDock(NULL)
{
	QFrame *vb = new QFrame(this);
	QVBoxLayout *layout = new QVBoxLayout(vb);
	layout->setMargin(0);
	vb->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
	vcht = new xChartView(this);
	tree = new xChartDatabase(this);
	//tree->bindItemComboBox(plot_item);
	prop = new xChartControl(this);
	//plot_item = new QComboBox(this);
	setMinimumSize(640, 580);
	resize(xSize, ySize);
	layout->addWidget(tree->plotItemComboBox());
	layout->addWidget(vcht);
	setCentralWidget(vb);

	prop->bindChartView(vcht);
	commDock = new QDockWidget(this);
	commDock->setWindowTitle("Command window");
	comm = new QLineEdit(this);
	commDock->setWidget(comm);
	//vcht->Chart()->axisX()->setTitleText(prop->LEs.at(LEXT)
	addDockWidget(Qt::RightDockWidgetArea, tree);
	addDockWidget(Qt::BottomDockWidgetArea, prop);
	addDockWidget(Qt::TopDockWidgetArea, commDock);
	mainToolBar = new QToolBar(this);
	addToolBar(mainToolBar);

	actions[WAVE_HEIGHT] = new QAction(QIcon(":/Resources/waveHeight.png"), tr("&Wave height"), this);
	actions[WAVE_HEIGHT]->setStatusTip(tr("Wave height"));
	connect(actions[WAVE_HEIGHT], SIGNAL(triggered()), this, SLOT(click_waveHeight()));
	connect(comm, SIGNAL(editingFinished()), this, SLOT(editingCommand()));
	connect(tree, SIGNAL(ClickedItem(int, QString)), this, SLOT(updateTargetItem(int, QString)));
	connect(tree->plotItemComboBox(), SIGNAL(currentIndexChanged(int)), this, SLOT(changeComboBoxItem(int)));
	mainToolBar->addAction(actions[WAVE_HEIGHT]);
	isActivate = true;
}

xChartWindow::~xChartWindow()
{
	if (tree) delete tree; tree = NULL;
	if (prop) delete prop; prop = NULL;
	if (comm) delete comm; comm = NULL;
	if (commDock) delete commDock; commDock = NULL;
	if (vcht) delete vcht; vcht = NULL;
	if (tree) delete tree; tree = NULL;

	//qDeleteAll(seriesMap);

	isActivate = false;
}

void xChartWindow::setChartData(xDynamicsManager* xdm)
{
	if (xdm)
	{
		if (xdm->XMBDModel())
		{
			tree->upload_mbd_results(xdm->XMBDModel());
		}
	}	
}

void xChartWindow::closeEvent(QCloseEvent *event)
{
	//isActivate = false;
}

void xChartWindow::editingCommand()
{
	QString com = comm->text();
	if (com.isEmpty())
		return;
	if (com == "manual property")
		isAutoUpdateProperties = false;
	else if (com == "auto property")
		isAutoUpdateProperties = true;
	else if (com == "stop update")
		isActivate = false;
	else if (com == "start update")
		isActivate = true;

	if (commandStatus == 1)
	{
		QStringList ss = comm->text().split("~");
		if (ss.size() != 2)
			return;
		waveHeightInputData.begin = ss.at(0).toInt();
		waveHeightInputData.end = ss.at(1).toInt();
		commandStatus = 2;
		comm->setText("");
		click_waveHeight();

	}
	else if (commandStatus == 3)
	{
		waveHeightInputData.location = comm->text().toDouble();
		commandStatus = 4;
		comm->setText("");
		click_waveHeight();
		return;
	}
	comm->setText("");
}

// void xChartWindow::setResultStorage(xChartDatabase* _rs)
// {
// 	//rs = _rs;
// }

void xChartWindow::uploadingResults()
{
	//cmdWindow::write(CMD_INFO, "The number of result parts : " + QString("%1").arg(model::rs->partList().size()));
// 	foreach(QString str, model::rs->partList())
// 	{
// 		tree->addChild(xChartDatabase::PART_ROOT, str);
// 	}
// 	//QStringList rlist = sph_model::SPHModel()->Sensors().keys();
// 	foreach(QString str, rlist)
// 	{
// 		tree->addChild(xChartDatabase::SENSOR_ROOT, str);
// 	}
// 	//rlist = model::rs->pointMassResults().keys();
// 	foreach(QString str, rlist)
// 	{
// 		tree->addChild(xChartDatabase::PMASS_ROOT, str);
// 	}
// 	//rlist = model::rs->reactionForceResults().keys();
// 	foreach(QString str, rlist)
// 	{
// 		tree->addChild(xChartDatabase::REACTION_ROOT, str);
// 	}
}

void xChartWindow::joint_plot()
{
	int it = tree->plotItemComboBox()->currentIndex();
	// 	QString target = tree->plotTarget();
	QString plotName = select_item_name + "_" + tree->plotItemComboBox()->currentText();
	if (!curPlotName.isEmpty())
		seriesMap[curPlotName]->hide();
	curPlotName = plotName;
	QLineSeries* series = seriesMap[plotName];
	if (series)
	{
		series->show();
		vcht->setCurrentLineSeries(series);
		prop->setPlotProperty(plotName);
		return;
	}
	series = createLineSeries(plotName);
	vcht->setCurrentLineSeries(series);
	double et = 0.0;
	double max_v = -FLT_MAX;
	double min_v = FLT_MAX;
	QString ytitle;
	QVector<xKinematicConstraint::kinematicConstraint_result>* pmr = tree->JointResults()[select_item_name];
	foreach(xKinematicConstraint::kinematicConstraint_result r, *pmr)
	{
		double v = 0.0;
		switch (it - 1)
		{
		case 0: v = r.location.x; ytitle = "Location(m)"; break;
		case 1: v = r.location.y; ytitle = "Location(m)"; break;
		case 2: v = r.location.z; ytitle = "Location(m)"; break;
		case 3: v = r.iaforce.x; ytitle = "Force(N)"; break;
		case 4: v = r.iaforce.y; ytitle = "Force(N)"; break;
		case 5: v = r.iaforce.z; ytitle = "Force(N)"; break;
		case 6: v = r.irforce.x; ytitle = "Torque(Nm)"; break;
		case 7: v = r.irforce.y; ytitle = "Torque(Nm)"; break;
		case 8: v = r.irforce.z; ytitle = "Torque(Nm)"; break;
		case 9: v = r.jaforce.x; ytitle = "Force(N)"; break;
		case 10: v = r.jaforce.y; ytitle = "Force(N)"; break;
		case 11: v = r.jaforce.z; ytitle = "Force(N)"; break;
		case 12: v = r.jrforce.x; ytitle = "Torque(Nm)"; break;
		case 13: v = r.jrforce.y; ytitle = "Torque(Nm)"; break;
		case 14: v = r.jrforce.z; ytitle = "Torque(Nm)"; break;
		}
		series->append(r.time, v);
		et = r.time;
		if (max_v < v) max_v = v;
		if (min_v > v) min_v = v;
	}
	double dy = (max_v - min_v) * 0.01;
	prop->setPlotProperty(plotName, "Time(sec)", ytitle, 0, et, min_v - dy, max_v + dy);
// 	QStringList sLists = tree->selectedLists();
// 	QString plotItem = tree->plotItemComboBox()->currentText();
// 	QString plotName;
// 	if (plotItem == "Wave height")
// 	{
// 		double t = 0;
// 		foreach(QString str, sLists)
// 		{
// 			sensor *s = sph_model::SPHModel()->Sensors()[str];
// 			double stime = s->samplingTime();
// 			plotName = s->Name() + "_" + plotItem;
// 			QLineSeries* series = createLineSeries(plotName);
// 			unsigned int cnt = 0;
// 			foreach(double d, s->scalarData())
// 			{
// 				t = cnt * stime;
// 				series->append(t, d);
// 				cnt++;
// 			}
// 		}
// 		prop->setPlotProperty(plotName, "Time(sec)", "Wave height(m)", 0, t, 0, 1.0);
// 		// 		vcht->Chart()->axisX()->setTitleText("Time(sec)");
// 		// 		vcht->Chart()->axisY()->setTitleText("Wave height(m)");
// 		// 		vcht->setAxisRange(0, t, 0, 1.0, true);
// 	}
}

void xChartWindow::body_plot()
{
 	int it = tree->plotItemComboBox()->currentIndex();
// 	QString target = tree->plotTarget();
 	QString plotName = select_item_name + "_" + tree->plotItemComboBox()->currentText();
 	if (!curPlotName.isEmpty())
 		seriesMap[curPlotName]->hide();
 	curPlotName = plotName;
 	QLineSeries* series = seriesMap[plotName];
 	if (series)
 	{
 		series->show();
		vcht->setCurrentLineSeries(series);
 		prop->setPlotProperty(plotName);
 		return;
	}
 	series = createLineSeries(plotName);
	vcht->setCurrentLineSeries(series);
 	double et = 0.0;
 	double max_v = -FLT_MAX;
 	double min_v = FLT_MAX;
 	QString ytitle;
	QVector<xPointMass::pointmass_result>* pmr = tree->MassResults()[select_item_name];
	foreach(xPointMass::pointmass_result r, *pmr)
	{
		double v = 0.0;
		switch (it-1)
		{
		case 0: v = r.pos.x; ytitle = "Position(m)"; break;
		case 1: v = r.pos.y; ytitle = "Position(m)"; break;
		case 2: v = r.pos.z; ytitle = "Position(m)"; break;
		case 3: v = r.vel.x; ytitle = "Velocity(m/s)"; break;
		case 4: v = r.vel.y; ytitle = "Velocity(m/s)"; break;
		case 5: v = r.vel.z; ytitle = "Velocity(m/s)"; break;
		case 6: v = r.omega.x; ytitle = "Ang. Velocity(rad/s)"; break;
		case 7: v = r.omega.y; ytitle = "Ang. Velocity(rad/s)"; break;
		case 8: v = r.omega.z; ytitle = "Ang. Velocity(rad/s)"; break;
		case 9: v = r.acc.x; ytitle = "Acceleration(m/s^2)"; break;
		case 10: v = r.acc.y; ytitle = "Acceleration(m/s^2)"; break;
		case 11: v = r.acc.z; ytitle = "Acceleration(m/s^2)"; break;
		case 12: v = r.alpha.x; ytitle = "Ang. Acceleration(rad/s^2)"; break;
		case 13: v = r.alpha.y; ytitle = "Ang. Acceleration(rad/s^2)"; break;
		case 14: v = r.alpha.z; ytitle = "Ang. Acceleration(rad/s^2)"; break;
		}
		series->append(r.time, v);
		et = r.time;
		if (max_v < v) max_v = v;
		if (min_v > v) min_v = v;
 	}
 	double dy = (max_v - min_v) * 0.01;
 	prop->setPlotProperty(plotName, "Time(sec)", ytitle, 0, et, min_v - dy, max_v + dy);
// 	//	else if (plotItem == )
}

//void xChartWindow::updatePlot(int root_id, QString selected_item)
//{
	
// 	QLineSeries* series = seriesMap[curPlotName];
// 	if (!series)
// 		return;
// 	QString target = tree->plotTarget();
// 	QList<resultStorage::pointMassResultData> pmr_list = model::rs->pointMassResults()[target];
// 	unsigned int s = pmr_list.size();
// 	resultStorage::pointMassResultData pmr = pmr_list.at(s - 1);
// 	int plotItem = tree->plotItemComboBox()->currentIndex();
// 	double v = 0;
// 	double et = 0.0;
// 	double max_v = 0.0;
// 	double min_v = 0.0;
// 	switch (plotItem)
// 	{
// 	case 0: v = pmr.pos.x; break;
// 	case 1: v = pmr.pos.y; break;
// 	case 2: v = pmr.pos.z; break;
// 	case 3: v = pmr.vel.x; break;
// 	case 4: v = pmr.vel.y; break;
// 	case 5: v = pmr.vel.z; break;
// 	case 6: v = pmr.omega.x; break;
// 	case 7: v = pmr.omega.y; break;
// 	case 8: v = pmr.omega.z; break;
// 	case 9: v = pmr.acc.x; break;
// 	case 10: v = pmr.acc.y; break;
// 	case 11: v = pmr.acc.z; break;
// 	case 12: v = pmr.alpha.x; break;
// 	case 13: v = pmr.alpha.y; break;
// 	case 14: v = pmr.alpha.z; break;
// 	}
// 	series->append(pmr.time, v);
// 	et = pmr.time;
// 	plotControlData pcd = prop->getPlotProperty(curPlotName);
// 	if (pcd.ymax < v) max_v = v;
// 	if (pcd.ymin > v) min_v = v;
// 	double dy = (max_v - min_v) * 0.01;
// 	//prop->setPlotProperty(curPlotName, pcd.xt, pcd.yt, 0, et, min_v - dy, max_v + dy);
// 
// 	//vcht->Chart()->update();
//}

void xChartWindow::updateTargetItem(int id, QString n)
{
	select_item_index = id;
	select_item_name = n;
}

void xChartWindow::click_waveHeight()
{
// 	whs = new waveHeightSensor;
// 		if (commandStatus == 0)
// 		{
// 			commDock->setWindowTitle("Input range of result part. (ex. begin~end)");
// 			commandStatus = 1;
// 		}
// 		else if (commandStatus == 2)
// 		{
// 			commDock->setWindowTitle("Input location.");
// 			commandStatus = 3;
// 		}
// 		else if (commandStatus == 4)
// 		{
// 			waveHeightSensor whs;
// 			double time, data;
// 			QLineSeries* series = createLineSeries("wave height");
// 			for (unsigned int i = waveHeightInputData.begin; i < waveHeightInputData.end; i++)
// 			{
// 				data = whs.measureByInputData(model::rs->getPartPosition(i), waveHeightInputData.location, model::rs->NFluid());
// 				time = model::rs->getPartTime(i);
// 				unsigned int cnt = 0;
// 				series->append(time, data);
// 			}
// 			prop->setPlotProperty("WaveHeight", "Time(sec)", "Wave height(m)", 0, time, 0, 1.0);
// 		}
}

void xChartWindow::changeComboBoxItem(int idx)
{
	switch (select_item_index)
	{
	case xChartDatabase::MASS_ROOT:
		body_plot();
		break;
	case xChartDatabase::KCONSTRAINT_ROOT:
		joint_plot();
		break;
	}
// 	//QString target = sLists.at(0);
// 	xChartDatabase::tRoot tp = tree->selectedType();
// 	switch (tp)
// 	{
// 	case xChartDatabase::SENSOR_ROOT:
// 		sensorItemPlot();
// 		break;
// 	case xChartDatabase::PMASS_ROOT:
// 		pointMassItemPlot();
// 		break;
// 	}
}

QLineSeries* xChartWindow::createLineSeries(QString n)
{
	QLineSeries* lineSeries = new QLineSeries;
	lineSeries->setName(n);
	vcht->addSeries(lineSeries);
	seriesMap[n] = lineSeries;
	return lineSeries;
}
