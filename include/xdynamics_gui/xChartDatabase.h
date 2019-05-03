#ifndef XCHARTDATABASE_H
#define XCHARTDATABASE_H

#include <QDockWidget>
#include <QTreeWidget>
#include <QComboBox>

class xMultiBodyModel;
class xPointMass;

class xChartDatabase : public QDockWidget
{
public:
	enum tRoot { PART_ROOT = 0, SENSOR_ROOT, PMASS_ROOT, REACTION_ROOT };
	xChartDatabase(QWidget* parent);
	~xChartDatabase();

	void addChild(tRoot, QString& _nm);
	void bindItemComboBox(QComboBox* t);
	QStringList selectedLists();
	tRoot selectedType();
	QComboBox* plotItemComboBox();
	QString plotTarget();
	void upload_mbd_results(xMultiBodyModel* xmbd);

	private slots:
	void contextMenu(const QPoint&);
	void clickItem(QTreeWidgetItem*, int);

private:
	QTreeWidget *tree;
	QMap<tRoot, QTreeWidgetItem*> roots;
	//QMap<QString, xPointMass::pointmass_result*> mbd_results;
	QStringList sLists;
	tRoot tSelected;
	QComboBox *plot_item;
	QString target;
};

#endif