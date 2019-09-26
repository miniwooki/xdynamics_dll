#ifndef XCHARTWINDOW_H
#define XCHARTWINDOW_H

#include <QMainWindow>
#include "xChartView.h"
//#include <QColumnView>
#include "xChartDatabase.h"
#include "xChartControl.h"
#include "xLineSeries.h"
//#include "resultStorage.h"

//class model;
class QStandardItem;
class QStandardItemModel;
class xCallOut;
class QToolBar;
class QLineEdit;
class xResultManager;
//class waveHeightSensor;

class xChartWindow : public QMainWindow
{
	Q_OBJECT

	enum ACTION_ID { PASSING_DISTRIBUTION = 0 };

	struct
	{
		int begin;
		int end;
		double location;
	}waveHeightInputData;

	//struct
	//{
	//	double 
	//	double area_y;
	//}passing_distribution_data;

public:
	xChartWindow(QWidget* parent = NULL);
	virtual ~xChartWindow();

	//void setResultStorage(resultStorage* _rs);
	bool setChartData(xResultManager* xdm);
	void joint_plot();
	void body_plot();
	void closeEvent(QCloseEvent *event);

	static bool IsActivate() { return isActivate; }
	static bool isActivate;

	public slots:
	void updateTargetItem(int, QString);

	private slots:
	void click_passing_distribution();
	void changeComboBoxItem(int i = 0);
	void editingCommand();

private:
	xLineSeries* createLineSeries(QString n);

private:
	bool isAutoUpdateProperties;
	bool isEditingCommand;
	int commandStatus;
	int wWidth;
	int wHeight;
	int xSize;
	int ySize;
	int openColumnCount;
	int select_item_index;
	QString select_item_name;
	QString curPlotName;
	xChartView *vcht;
	xChartDatabase *tree;
	xChartControl *prop;
	QStandardItemModel *cmodel;
	xCallOut *m_tooltip;
	QList<xCallOut *> m_callouts;
	QToolBar *mainToolBar;
	QMap<ACTION_ID, QAction*> actions;
	QMap<QString, xLineSeries*> seriesMap;
	QDockWidget *commDock;
	QLineEdit *comm;
};
#endif