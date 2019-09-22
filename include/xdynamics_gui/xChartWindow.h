#ifndef XCHARTWINDOW_H
#define XCHARTWINDOW_H

#include <QMainWindow>
#include "xChartView.h"
//#include <QColumnView>
#include "xChartDatabase.h"
#include "xChartControl.h"
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
	void uploadingResults();
	void joint_plot();
	void body_plot();
	void closeEvent(QCloseEvent *event);

	static bool IsActivate() { return isActivate; }
	static bool isActivate;

	public slots:
	void updateTargetItem(int, QString);

	private slots:
	void click_passing_distribution();
	void changeComboBoxItem(int);
	void editingCommand();

private:
	QLineSeries* createLineSeries(QString n);
	//void columnViewAdjustSameSize(int ncol);
	//void setBodyRoot(QStandardItem *p);
	/*void setJointForceRoot(QStandardItem *p);*/
	//void resizeEvent(QResizeEvent *event);

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
	//resultStorage* rs;
	xCallOut *m_tooltip;
	QList<xCallOut *> m_callouts;
	QToolBar *mainToolBar;
	QMap<ACTION_ID, QAction*> actions;
	//QComboBox* plot_item;
	QMap<QString, QLineSeries*> seriesMap;
	QDockWidget *commDock;
	QLineEdit *comm;
	//waveHeightSensor *whs;
	//xResultManager* xrm;


};
#endif