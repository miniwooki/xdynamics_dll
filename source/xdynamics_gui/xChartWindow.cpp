#include "xChartWindow.h"
//#include "model.h"
//#include "waveHeightSensor.h"
#include "xCallOut.h"
//#include "cmdWindow.h"
#include "xdynamics_manager/xResultManager.h"
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
	, comm(NULL)
	, commDock(NULL)
	, plot_item(NULL)
{
	QFrame *vb = new QFrame(this);
	QVBoxLayout *layout = new QVBoxLayout(vb);
	layout->setMargin(0);
	vb->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
	vcht = new xChartView(this);
	tree = new xChartDatabase(this);
	prop = new xChartControl(this);
	plot_item = new QComboBox(this);
	setMinimumSize(640, 580);
	resize(xSize, ySize);
	layout->addWidget(plot_item);
	layout->addWidget(vcht);
	setCentralWidget(vb);

	prop->bindChartView(vcht);
	commDock = new QDockWidget(this);
	commDock->setWindowTitle("Command window");
	comm = new QLineEdit(this);
	commDock->setWidget(comm);
	addDockWidget(Qt::RightDockWidgetArea, tree);
	addDockWidget(Qt::BottomDockWidgetArea, prop);
	addDockWidget(Qt::TopDockWidgetArea, commDock);
	mainToolBar = new QToolBar(this);
	addToolBar(mainToolBar);

	connect(comm, SIGNAL(editingFinished()), this, SLOT(editingCommand()));
	connect(tree, SIGNAL(ClickedItem(int, QString, QStringList)), this, SLOT(updateTargetItem(int, QString, QStringList)));
	//connect(tree, &QTreeWidget::itemClicked, this, &xChartDatabase::clickItem);
	connect(plot_item, SIGNAL(currentIndexChanged(int)), this, SLOT(PlotFromComboBoxItem(int)));
	QShortcut *a = new QShortcut(QKeySequence("Shift+R"), this);
	connect(a, SIGNAL(activated()), this, SLOT(changeComboBoxItem()));
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

	isActivate = false;
}

bool xChartWindow::setChartData(xResultManager* xrm)
{
	if (xrm)
	{
		tree->setResultManager(xrm);
		return false;
	}		
	return true;
}

void xChartWindow::closeEvent(QCloseEvent *event)
{

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

	}
	else if (commandStatus == 3)
	{
		waveHeightInputData.location = comm->text().toDouble();
		commandStatus = 4;
		comm->setText("");
		return;
	}
	comm->setText("");
}

void xChartWindow::joint_plot()
{
	int it = plot_item->currentIndex();
	QString plotName = select_item_name + "_" + plot_item->currentText();
	if (!curPlotName.isEmpty())
		seriesMap[curPlotName]->hide();
	curPlotName = plotName;
	xLineSeries* series = seriesMap[plotName];
	if (series)
	{
		series->show();
		vcht->setCurrentLineSeries(series);
		prop->setPlotProperty(plotName);
	}
	else
	{
		series = createLineSeries(plotName);
		vcht->setCurrentLineSeries(series);
	}
	double et = series->get_end_time();
	double max_v = -FLT_MAX;
	double min_v = FLT_MAX;
	QString ytitle;
	xKinematicConstraint::kinematicConstraint_result* pmr = tree->JointResults(select_item_name);
	double *time = tree->result_manager_ptr()->get_times();
	unsigned int npt = series->get_num_point();
	for(unsigned int i = npt; i < tree->result_manager_ptr()->get_num_parts(); i++)
	{
		xKinematicConstraint::kinematicConstraint_result r = pmr[i];
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
		series->append(time[i], v);
		et = time[i];
		series->set_end_time(et);
		series->set_num_point(i);
		if (max_v < v) max_v = v;
		if (min_v > v) min_v = v;
	}
	double dy = (max_v - min_v) * 0.01;
	prop->setPlotProperty(plotName, "Time(sec)", ytitle, 0, et, min_v - dy, max_v + dy);
}

void xChartWindow::body_plot()
{
 	int it = plot_item->currentIndex();
// 	QString target = tree->plotTarget();
 	QString plotName = select_item_name + "_" + plot_item->currentText();
 	if (!curPlotName.isEmpty())
 		seriesMap[curPlotName]->hide();
 	curPlotName = plotName;
 	xLineSeries* series = seriesMap[plotName];
 	if (series)
 	{
 		series->show();
		vcht->setCurrentLineSeries(series);
 		prop->setPlotProperty(plotName);
 		//return;
	}
	else
	{
		series = createLineSeries(plotName);
		vcht->setCurrentLineSeries(series);
	} 	
 	double et = series->get_end_time();
 	double max_v = -FLT_MAX;
 	double min_v = FLT_MAX;
 	QString ytitle;
	xPointMass::pointmass_result* pmr = tree->MassResults(select_item_name);
	double *time = tree->result_manager_ptr()->get_times();
	unsigned int npt = series->get_num_point();
	for (unsigned int i = series->get_num_point(); i < tree->result_manager_ptr()->get_current_part_number(); i++)
	{
		xPointMass::pointmass_result r = pmr[i];
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
		series->append(time[i], v);
		et = time[i];// r.time;
		series->set_end_time(et);
		series->set_num_point(i);
		if (max_v < v) max_v = v;
		if (min_v > v) min_v = v;
 	}
 	double dy = (max_v - min_v) * 0.01;
 	prop->setPlotProperty(plotName, "Time(sec)", ytitle, 0, et, min_v - dy, max_v + dy);
// 	//	else if (plotItem == )
}

void xChartWindow::updateTargetItem(int id, QString n, QStringList plist)
{
	select_item_index = id;
	select_item_name = n;
	plot_item->clear();
	plot_item->addItems(plist);
}

void xChartWindow::click_passing_distribution()
{

}

void xChartWindow::PlotFromComboBoxItem(int idx)
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
}

xLineSeries* xChartWindow::createLineSeries(QString n)
{
	xLineSeries* lineSeries = new xLineSeries;
	lineSeries->setName(n);
	vcht->addSeries(lineSeries);
	seriesMap[n] = lineSeries;
	return lineSeries;
}
