#ifndef XCHARTDATABASE_H
#define XCHARTDATABASE_H

#include <QDockWidget>
#include <QTreeWidget>
#include <QComboBox>
#include "xdynamics_manager/xMultiBodyModel.h"

inline QStringList get_point_mass_chart_list()
{
	QStringList stList; stList.push_back("None");
	stList.push_back("PX"); stList.push_back("PY"); stList.push_back("PZ");
	stList.push_back("VX"); stList.push_back("VY"); stList.push_back("VZ");
	stList.push_back("RVX"); stList.push_back("RVY"); stList.push_back("RVZ");
	stList.push_back("AX"); stList.push_back("AY"); stList.push_back("AZ");
	stList.push_back("RAX"); stList.push_back("RAY"); stList.push_back("RAZ");
	return stList;
}

inline QStringList get_joint_chart_list()
{
	QStringList stList; stList.push_back("None");
	stList.push_back("LX"); stList.push_back("LY"); stList.push_back("LZ");
	stList.push_back("IFX"); stList.push_back("IFY"); stList.push_back("IFZ");
	stList.push_back("ITX"); stList.push_back("ITY"); stList.push_back("ITZ");
	stList.push_back("JFX"); stList.push_back("JFY"); stList.push_back("JFZ");
	stList.push_back("JTX"); stList.push_back("JTY"); stList.push_back("JTZ");
	return stList;
}

inline QStringList get_particle_chart_list()
{
	QStringList stList; stList.push_back("None");
	stList.push_back("PX"); stList.push_back("PY"); stList.push_back("PZ");
	stList.push_back("VX"); stList.push_back("VY"); stList.push_back("VZ");
	stList.push_back("RVX"); stList.push_back("RVY"); stList.push_back("RVZ");

	return stList;
}

class xChartDatabase : public QDockWidget
{
	Q_OBJECT

public:
	enum tRoot { MASS_ROOT = 0, KCONSTRAINT_ROOT, PARTICLE_ROOT };
	xChartDatabase(QWidget* parent);
	~xChartDatabase();

	void addChild(tRoot, QString& _nm);
	void bindItemComboBox(QComboBox* t);
	QStringList selectedLists();
	tRoot selectedType();
	QComboBox* plotItemComboBox();
	QString plotTarget();
	void upload_mbd_results(xMultiBodyModel* xmbd);
	QMap<QString, QVector<xPointMass::pointmass_result>*>& MassResults() { return mass_results; }
	QMap<QString, QVector<xKinematicConstraint::kinematicConstraint_result>*>& JointResults() { return constraint_results; }

	private slots:
	void contextMenu(const QPoint&);
	void clickItem(QTreeWidgetItem*, int);
	void selectPlotItem(int);

private:
	void export_to_textfile(QTreeWidgetItem* citem);
	void process_context_menu(QString txt, QTreeWidgetItem* citem);
	QTreeWidget *tree;
	QTreeWidgetItem* current_item;
	QMap<tRoot, QTreeWidgetItem*> roots;
	QMap<QString, QVector<xPointMass::pointmass_result>*> mass_results;
	QMap<QString, QVector<xKinematicConstraint::kinematicConstraint_result>*> constraint_results;
	QStringList sLists;
	tRoot tSelected;
	QComboBox *plot_item;
	QString target;

signals:
	void ClickedItem(int, QString);
};

#endif