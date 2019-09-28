#include "xChartDatabase.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/xResultManager.h"
///#include "cmdWindow.h"
#include "xmap.hpp"
#include <QAction>
#include <QMenu>
#include <QDebug>

xChartDatabase::xChartDatabase(QWidget* parent)
	: QDockWidget(parent)
	, time(NULL)
{
	setWindowTitle("Result database");
	tree = new QTreeWidget;
	tree->setColumnCount(1);
	tree->setContextMenuPolicy(Qt::CustomContextMenu);
	roots[MASS_ROOT] = new QTreeWidgetItem(tree);
	roots[KCONSTRAINT_ROOT] = new QTreeWidgetItem(tree);
	roots[PARTICLE_ROOT] = new QTreeWidgetItem(tree);
// 	roots[PART_ROOT] = new QTreeWidgetItem(tree);
// 	roots[SENSOR_ROOT] = new QTreeWidgetItem(tree);
// 	roots[PMASS_ROOT] = new QTreeWidgetItem(tree);
// 	roots[REACTION_ROOT] = new QTreeWidgetItem(tree);
// 	//roots[CUBE_ROOT] = new QTreeWidgetItem(vtree);
// 	//roots[CYLINDER_ROOT] = new QTreeWidgetItem(vtree);
// 	//roots[POLYGON_ROOT] = new QTreeWidgetItem(vtree);
// 	//roots[MASS_ROOT] = new QTreeWidgetItem(vtree);
// 
	roots[MASS_ROOT]->setText(0, "Body");
	roots[KCONSTRAINT_ROOT]->setText(0, "Joint");
	roots[PARTICLE_ROOT]->setText(0, "Particle");
// 	roots[PART_ROOT]->setText(0, "Part");
// 	roots[SENSOR_ROOT]->setText(0, "Sensor");
// 	roots[PMASS_ROOT]->setText(0, "Mass");
// 	roots[REACTION_ROOT]->setText(0, "Reaction");
// 	/*roots[CUBE_ROOT]->setText(0, "Cube");
// 	roots[CYLINDER_ROOT]->setText(0, "Cylinder");
// 	roots[POLYGON_ROOT]->setText(0, "Polygon");
// 	roots[MASS_ROOT]->setText(0, "Mass");
// 	*/
// 	roots[PART_ROOT]->setIcon(0, QIcon(":/Resources/parts-256.png"));
// 	roots[SENSOR_ROOT]->setIcon(0, QIcon(":/Resources/Sensor_icon.png"));
// 	roots[PMASS_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));
// 	roots[REACTION_ROOT]->setIcon(0, QIcon(":/Resources/revolute.png"));
	/*roots[CUBE_ROOT]->setIcon(0, QIcon(":/Resources/pRec.png"));
	roots[CYLINDER_ROOT]->setIcon(0, QIcon(":/Resources/cylinder.png"));
	roots[POLYGON_ROOT]->setIcon(0, QIcon(":/Resources/icPolygon.png"));
	roots[MASS_ROOT]->setIcon(0, QIcon(":/Resources/mass.png"));*/
	//plot_item = new QComboBox;
	connect(tree, &QTreeWidget::customContextMenuRequested, this, &xChartDatabase::contextMenu);
	connect(tree, &QTreeWidget::itemClicked, this, &xChartDatabase::clickItem);
	//connect(&plot_item, SIGNAL(currentIndexChanged(int)), this, SLOT(selectPlotItem(int)));

	tree->setSelectionMode(QAbstractItemView::SelectionMode::ContiguousSelection);
	setWidget(tree);
}

// xChartDatabase::~xChartDatabase()
// {
// 	
// }

void xChartDatabase::contextMenu(const QPoint& pos)
{
	QTreeWidgetItem* item = tree->itemAt(pos);
	if (!item->parent())
		return;
	current_item = item;
	QAction *act = new QAction(tr("Property"), this);
	QAction *export_txt = new QAction(tr("Export text file"), this);
	act->setStatusTip(tr("property menu"));
	export_txt->setStatusTip(tr("Export to text file"));
	connect(export_txt, SIGNAL(triggered()), this, SLOT(act_export_to_textfile()));
	//nonnect(act, SIGNAL(triggered()), this, SLOT(actProperty()));

	QMenu menu(this);
	menu.addAction(act);
	menu.addAction(export_txt);

	QPoint pt(pos);
	QAction *selectedMenu = menu.exec(tree->mapToGlobal(pos));
	if(selectedMenu)
		process_context_menu(selectedMenu->text(), item);
}

void xChartDatabase::clickItem(QTreeWidgetItem* item, int col)
{
	if (!item)
		return;

	//plot_item.clear();
	//QList<QTreeWidgetItem*> items = tree->selectedItems();
	//cmdWindow::write(CMD_INFO, "The number of selected items : " + QString("%1").arg(items.size()));
// 	if (items.size() > 1)
// 	{
// 		foreach(QTreeWidgetItem* it, items)
// 			sLists.push_back(it->text(col));
// 		return;
// 	}
	QString tg = item->text(col);
	//qDebug() << "target_name : " << tg;
// 	if (target == tg)
// 		return;
// 	target = tg;
	QTreeWidgetItem* parent = item->parent();
	if (!parent)
	{
		return;
	}
	QString sp = parent->text(0);
	//qDebug() << "parent_name : " << sp;
	int idx = -1;
	QStringList plist;
	if (sp == "Body")
	{
		idx = (int)MASS_ROOT;
		//qDebug() << "before add items";
		plist = get_point_mass_chart_list();
		//qDebug() << "after add items";
	//	qDebug() << "fdljksdfkljsdfkljsdf";
	}
	else if (sp == "Joint")
	{
		idx = (int)KCONSTRAINT_ROOT;
		plist = get_joint_chart_list();
	}
	else if (sp == "Particle")
	{
		idx = (int)PARTICLE_ROOT;
		plist = get_particle_chart_list();
	}
	emit ClickedItem(idx, tg, plist);
	//plot_item.setCurrentIndex(0);
// 	{
// // 		tSelected = SENSOR_ROOT;
// // 		sLists.push_back(target);
// // 		plot_item->clear();
// // 		sensor* s = sph_model::SPHModel()->Sensors()[target];
// // 		switch (s->sensorType())
// // 		{
// // 		case sensor::WAVE_HEIGHT_SENSOR:
// // 			plot_item->addItem("Wave height");
// // 			break;
// // 		}
// 	}
// 	else if (sp == "Mass")
// 	{
// // 		tSelected = PMASS_ROOT;
// // 		plot_item->clear();
// // 		//plot_item->addItem("PX");
// // 		plot_item->addItems(getPMResultString());
// 	}
// 	else if (sp == "Reaction")
// 	{
// // 		tSelected = REACTION_ROOT;
// // 		plot_item->clear();
// // 		plot_item->addItems(getRFResultString());
// 	}

}

void xChartDatabase::selectPlotItem(int id)
{
	emit ChangeComboBoxItem(id);
}

void xChartDatabase::export_to_textfile(QTreeWidgetItem* citem)
{
	QString cname = citem->text(0);
	if (citem->parent()->text(0) == "Body")
		xrm->export_mass_result_to_text(cname.toStdString());
	else if (citem->parent()->text(0) == "Joint")
		xrm->export_joint_result_to_text(cname.toStdString());
}

void xChartDatabase::process_context_menu(QString txt, QTreeWidgetItem* citem)
{
	//.QString txt = a->text();
	if (txt == "Export text file" && citem)
	{
		
		export_to_textfile(citem);
	}
}

xChartDatabase::~xChartDatabase()
{
	if (tree) delete tree; tree = NULL;
}

void xChartDatabase::addChild(tRoot tr, QString& _nm)
{
	QTreeWidgetItem* child = new QTreeWidgetItem();
	child->setText(0, _nm);
	roots[tr]->addChild(child);
	//cmdWindow::write(CMD_INFO, "Added root item : " + _nm);
}

void xChartDatabase::bindItemComboBox(QComboBox* t)
{
//plot_item = t;
}

QStringList xChartDatabase::selectedLists()
{
	return sLists;
}

xChartDatabase::tRoot xChartDatabase::selectedType()
{
	return tSelected;
}

void xChartDatabase::setResultManager(xResultManager * _xrm)
{
	xrm = _xrm;
	xmap<xstring, struct_pmr*>* pmrs = xrm->get_mass_result_xmap();
	xmap<xstring, struct_kcr*>* kcrs = xrm->get_joint_result_xmap();
	if (pmrs->size())
	{
		for (xmap<xstring, struct_pmr*>::iterator it = pmrs->begin(); it != pmrs->end(); it.next())
		{
			QString _nm = it.key().text();
			mass_results[_nm] = it.value();
			addChild(MASS_ROOT, _nm);
		}
	}
	if (kcrs->size())
	{
		for (xmap<xstring, struct_kcr*>::iterator it = kcrs->begin(); it != kcrs->end(); it.next())
		{
			QString _nm = it.key().text();
			constraint_results[_nm] = it.value();
			addChild(KCONSTRAINT_ROOT, _nm);
		}
	}
}

QString xChartDatabase::plotTarget()
{
	return target;
}

xResultManager * xChartDatabase::result_manager_ptr()
{
	return xrm;
}

void xChartDatabase::upload_mbd_results(xMultiBodyModel* _xmbd)
{
	//xmap<xstring, xPointMass*> *xxx = _xmbd->Masses_ptr();
	//xPointMass* xxx = _xmbd->XMass("");
	/*for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
	{
		xPointMass* xpm = it.value();
		mass_results[xpm->Name()] = xpm->XPointMassResultPointer();
		addChild(MASS_ROOT, xpm->Name());
	}
	for(xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
	{
		xKinematicConstraint* xkc = it.value();
		constraint_results[xkc->Name()] = xkc->XKinematicConstraintResultPointer();
		addChild(KCONSTRAINT_ROOT, xkc->Name());
	}*/
}

xPointMass::pointmass_result* xChartDatabase::MassResults(QString name)
{
	QMap<QString, xPointMass::pointmass_result*>::iterator it = mass_results.find(name);
	if (it != mass_results.end())
		return it.value();
	return NULL;
}

xKinematicConstraint::kinematicConstraint_result * xChartDatabase::JointResults(QString name)
{
	QMap<QString, xKinematicConstraint::kinematicConstraint_result*>::iterator it = constraint_results.find(name);
	if (it != constraint_results.end())
		return it.value();
	return NULL;
}
