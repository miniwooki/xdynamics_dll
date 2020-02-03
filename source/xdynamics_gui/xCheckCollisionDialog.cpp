#include "xCheckCollisionDialog.h"
#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xParticleMananger.h"
#include "xvParticle.h"

xCheckCollisionDialog::xCheckCollisionDialog(QWidget* parent)
	: QDialog(parent)
	, xp(nullptr)
	, pmgr(nullptr)
	, isSetup(false)
	, isChangedSelection(false)
{
	setupUi(this);
	connect(PB_Check, &QPushButton::clicked, this, &xCheckCollisionDialog::checkCollision);
	connect(CollisionParticle, &QTreeWidget::itemClicked, this, &xCheckCollisionDialog::clickTreeItem);
	connect(Information, &QTableWidget::cellClicked, this, &xCheckCollisionDialog::highlightSelectedCluster);
	connect(Information, &QTableWidget::itemSelectionChanged, this, &xCheckCollisionDialog::selectedItemProcess);
	connect(Information, &QTableWidget::itemChanged, this, &xCheckCollisionDialog::changePosition);
}

xCheckCollisionDialog::~xCheckCollisionDialog()
{

}

void xCheckCollisionDialog::selectedItemProcess()
{
	isChangedSelection = true;
}

void xCheckCollisionDialog::changePosition(QTableWidgetItem * item)
{
	if (!isSetup)
		return;
	if (!Information->rowCount())
		return;
	int row = Information->currentRow();
	int column = Information->currentColumn();
	xClusterInformation info;
	foreach(info, cinfos) {
		if (row >= info.sid && row * info.neach < info.sid + info.count)
			break;
	}
	unsigned int cid = info.sid + row * info.neach;
	for (unsigned int i = cid; i < cid + info.neach; i++) {
		QTableWidgetItem* x = Information->item(row, 0);
		QTableWidgetItem* y = Information->item(row, 1);
		QTableWidgetItem* z = Information->item(row, 2);
		xp->ChangePosition(
			i, 
			x->text().toDouble(), 
			y->text().toDouble(), 
			z->text().toDouble());
	}
}

void xCheckCollisionDialog::mouseReleaseEvent(QMouseEvent *event)
{
	/*if (isChangedSelection) {
		QList<QTableWidgetItem*> items = Information->selectedItems();

	}*/
}

void xCheckCollisionDialog::highlightSelectedCluster(int row, int column)
{
	if (changedColor.size()) {
		QMapIterator<unsigned int, QColor> color(changedColor);
		QColor previous_color;
		while (color.hasNext()) {
			color.next();
			xp->ChangeColor(color.key(), color.value(), previous_color);
		}
		changedColor.clear();
	}
	QColor previous_color;
	xClusterInformation info;
	foreach(info, cinfos) {
		if (row >= info.sid && row * info.neach < info.sid + info.count)
			break;
	}
	unsigned int cid = info.sid + row * info.neach;
	for (unsigned int i = cid; i < cid + info.neach; i++) {
		xp->ChangeColor(i, QColor(255, 0, 0), previous_color);
		changedColor[i] = previous_color;
	}
}

void xCheckCollisionDialog::clickTreeItem(QTreeWidgetItem * item, int column)
{
	if (changedColor.size()) {
		QMapIterator<unsigned int, QColor> color(changedColor);
		QColor previous_color;
		while (color.hasNext()) {
			color.next();
			xp->ChangeColor(color.key(), color.value(), previous_color);
		}
		changedColor.clear();
	}
	if (!item->parent()) {
		unsigned int id = item->text(0).mid(item->text(0).lastIndexOf("e")+1).toUInt();
		QColor previous_color;
		xp->ChangeColor(id, QColor(255, 0, 0), previous_color);
		changedColor[id] = previous_color;
		for (unsigned int i = 0; i < item->childCount(); i++) {
			QTreeWidgetItem* child = item->child(i);
			unsigned int jd = child->text(0).mid(child->text(0).lastIndexOf("e") + 1).toUInt();
			xp->ChangeColor(jd, QColor(0, 255, 0), previous_color);
			changedColor[jd] = previous_color;
		}
	}
	else if (item->parent()) {
		unsigned int id = item->text(0).mid(item->text(0).lastIndexOf("e") + 1).toUInt();
		QColor previous_color;
		xp->ChangeColor(id, QColor(255, 0, 0), previous_color);
		changedColor[id] = previous_color;
		QTreeWidgetItem* parent = item->parent();
		unsigned int jd = parent->text(0).mid(parent->text(0).lastIndexOf("e") + 1).toUInt();
		xp->ChangeColor(jd, QColor(0, 255, 0), previous_color);
		changedColor[jd] = previous_color;
	}
}

void xCheckCollisionDialog::setup(xvParticle* _xp, xParticleManager* _pmgr, xObjectManager* _omgr)
{
	xp = _xp;
	pmgr = _pmgr;
	omgr = _omgr;
	foreach(xvParticle::particleGroupData p, xp->ParticleGroupData()) {
		QListWidgetItem* item = new QListWidgetItem(p.name);
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(Qt::Checked);
		ParticleList->addItem(item);
		items.append(item);
	}

	unsigned int np = 0;
	unsigned int ncp = 0;
	QMap<QString, QString> cpo_pair;
	foreach(QListWidgetItem* item, items) {
		//if (item->checkState() == Qt::Checked) {
		xParticleObject* xpo = pmgr->XParticleObject(item->text().toStdString());
		xClusterInformation cinfo;
		
		if (xpo->NumCluster()) {
			QString poName = QString::fromStdString(xpo->Name());
			QString cName = QString::fromStdString(xpo->ParticleShapeName());
			cinfo.count = xpo->NumCluster();
			cinfo.sid = np;
			cinfos[poName] = cinfo;
			cpo_pair[cName] = poName;

		}
		np += xpo->NumParticle();
		ncp += xpo->NumCluster();
		//}
	}

	map<std::string, xClusterObject*> clusters = omgr->XClusterObjects();
	//xClusterInformation* _cinfos = new xClusterInformation[cinfos.size()];
	unsigned int count = 0;
	for (map<std::string, xClusterObject*>::iterator it = clusters.begin(); it != clusters.end(); it++) {		
		QString name = QString::fromStdString(it->second->Name());
		if (cpo_pair.find(name) == cpo_pair.end())
			continue;
		xClusterInformation info = cinfos[cpo_pair[name]];
		info.neach = it->second->NumElement();
		info.count *= info.neach;
		cinfos[cpo_pair[name]] = info;
		//_cinfos[count++] = info;
	}

//	QMap<QString, xClusterInformation> cinfos;


	double *pos = new double[np * 4];
	double *ep = new double[np * 4];
	//cpos = new double[ncp * 4];
	double *cpos = nullptr;
	if (ncp)
		cpos = new double[ncp * 4];
	pmgr->CopyPosition(pos, cpos, ep, np);
	QStringList labels = { "X", "Y", "Z" };
	Information->setColumnCount(3);
	Information->setHorizontalHeaderLabels(labels);
	if (cpos) {
		Information->setRowCount(ncp);
		for (unsigned int i = 0; i < ncp; i++) {
			QTableWidgetItem *item = nullptr;
			Information->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 0])));
			Information->setItem(i, 1, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 1])));
			Information->setItem(i, 2, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 2])));
			//Information->setItem(i, 3, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 3])));
		}
	}
	if(pos) delete[] pos;
	if(ep) delete[] ep;
	if(cpos) delete[] cpos;
	isSetup = true;
}

void xCheckCollisionDialog::checkCollision()
{
	//map<std::string, xClusterObject*> clusters = omgr->XClusterObjects();
	//
	//
	//for (map<std::string, xClusterObject*>::iterator it = clusters.begin(); it != clusters.end(); it++) {
	//	QString name = QString::fromStdString(it->second->Name());
	//	xClusterInformation info = cinfos[name];
	//	info.neach = it->second->NumElement();
	//	info.count *= info.neach;
	//	
	//}
	unsigned int count = 0;
	unsigned int np = 0;
	unsigned int ncp = 0;
	QList<xParticleObject*> pobjects;
	foreach(QListWidgetItem* item, items) {
		if (item->checkState() == Qt::Checked) {
			xParticleObject* xpo = pmgr->XParticleObject(item->text().toStdString());
			np += xpo->NumParticle();
			ncp += xpo->NumCluster();
			pobjects.append(xpo);
			count++;
		}
	}
	xClusterInformation* _cinfos = new xClusterInformation[count];
	count = 0;
	unsigned int sid = 0;
	foreach(QListWidgetItem* item, items) {
		if (item->checkState() == Qt::Checked) {
			xParticleObject* xpo = pmgr->XParticleObject(item->text().toStdString());
			_cinfos[count] = cinfos[item->text()];
			_cinfos[count++].sid = sid;
			sid += xpo->NumParticle();
		}
	}
	double *pos = new double[np * 4];
	double *ep = new double[np * 4];
	//cpos = new double[ncp * 4];
	double *cpos = nullptr;
	if (ncp)
		cpos = new double[ncp * 4];
	sid = 0;
	foreach(xParticleObject* xpo, pobjects) {
		xpo->CopyPosition(pos);
		if (cpos && xpo->ShapeForm() == CLUSTER_SHAPE) {
			xpo->CopyClusterPosition(sid, cpos, ep);
			sid += xpo->NumParticle();
		}			
	}
	double minRadius = FLT_MAX;
	double maxRadius = -FLT_MAX;
	for (unsigned int i = 0; i < np; i++) {
		double r = pos[i * 4 + 3];
		if (r > maxRadius) maxRadius = r;
		if (r < minRadius) minRadius = r;
	}
	xNeiborhoodCell dtor;
	dtor.setWorldOrigin(new_vector3d(-1.0, -1.0, -1.0));
	dtor.setGridSize(new_vector3ui(128, 128, 128));
	dtor.setCellSize(maxRadius * 2.0);
	dtor.initialize(np);
	dtor.detectionCpu(pos, np, 0);
	dtor.rearrange_cell();
	xContactManager cmgr;
	std::map<pair<unsigned int, unsigned int>, xPairData> c_pairs;
	c_pairs = cmgr.CalculateCollisionPair(
		(vector4d*)pos,
		dtor.sortedID(),
		dtor.cellStart(),
		dtor.cellEnd(),
		_cinfos,
		count,
		np);
	std::map<pair<unsigned int, unsigned int>, xPairData>::iterator it = c_pairs.begin();
	QString prefix = "Particle ";
	for (; it != c_pairs.end(); it++) {
		QString si = prefix + QString("%1").arg(it->first.first);
		QString sj = prefix + QString("%1").arg(it->first.second);

		if (parents.find(si) == parents.end()) {
			parents[si] = new QTreeWidgetItem(CollisionParticle);
			parents[si]->setText(0, si);
			
		//	parents[si]->setData(0, 0, QVariant(it->first.first));
		}
		QTreeWidgetItem* child = new QTreeWidgetItem(parents[si]);
		child->setText(0, sj);
		//child->setData(0, 0, QVariant(it->first.second));
	}
	if (pos) delete[] pos;
	if (ep) delete[] ep;
	if (cpos) delete[] cpos;
}