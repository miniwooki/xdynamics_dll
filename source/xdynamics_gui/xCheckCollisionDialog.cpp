#include "xCheckCollisionDialog.h"
#include "xdynamics_algebra/xNeiborhoodCell.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xParticleMananger.h"
#include "xvParticle.h"

#include <QTextStream>
#include <QMessageBox>

xCheckCollisionDialog::xCheckCollisionDialog(QWidget* parent)
	: QDialog(parent)
	, xp(nullptr)
	, pmgr(nullptr)
	, isSetup(false)
	, isChangedSelection(false)
	, selectedCluster(-1)
{
	setupUi(this);
	connect(PB_Check, &QPushButton::clicked, this, &xCheckCollisionDialog::checkCollision);
	connect(CollisionParticle, &QTreeWidget::itemClicked, this, &xCheckCollisionDialog::clickTreeItem);
	connect(PlusX, &QPushButton::clicked, this, &xCheckCollisionDialog::movePlusX);
	connect(PlusY, &QPushButton::clicked, this, &xCheckCollisionDialog::movePlusY);
	connect(PlusZ, &QPushButton::clicked, this, &xCheckCollisionDialog::movePlusZ);
	connect(MinusX, &QPushButton::clicked, this, &xCheckCollisionDialog::moveMinusX);
	connect(MinusY, &QPushButton::clicked, this, &xCheckCollisionDialog::moveMinusY);
	connect(MinusZ, &QPushButton::clicked, this, &xCheckCollisionDialog::moveMinusZ);
	connect(PlusNormal, &QPushButton::clicked, this, &xCheckCollisionDialog::movePlusNormal);
	connect(MinusNormal, &QPushButton::clicked, this, &xCheckCollisionDialog::moveMinusNormal);
	//connect(Information, &QTableWidget::cellClicked, this, &xCheckCollisionDialog::highlightSelectedCluster);
	//connect(Information, &QTableWidget::itemSelectionChanged, this, &xCheckCollisionDialog::selectedItemProcess);
	//connect(Information, &QTableWidget::itemChanged, this, &xCheckCollisionDialog::changePosition);
}

xCheckCollisionDialog::~xCheckCollisionDialog()
{

}

void xCheckCollisionDialog::selectedItemProcess()
{
	isChangedSelection = true;
}

void xCheckCollisionDialog::movePlusX()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, length, 0, 0);
	}
	updateCollision();
}

void xCheckCollisionDialog::movePlusY()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, 0, length, 0);
	}
	updateCollision();
}

void xCheckCollisionDialog::movePlusZ()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, 0, 0, length);
	}
	updateCollision();
}

void xCheckCollisionDialog::moveMinusX()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, -length, 0, 0);
	}
	updateCollision();
}

void xCheckCollisionDialog::moveMinusY()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, 0, -length, 0);
	}
	updateCollision();
}

void xCheckCollisionDialog::moveMinusZ()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach, 0, 0, -length);
	}
	updateCollision();
}

void xCheckCollisionDialog::movePlusNormal()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	QStringList normal = LE_Direction->text().split(",");
	double ux = normal.at(0).toDouble();
	double uy = normal.at(1).toDouble();
	double uz = normal.at(2).toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach + i, length * ux, length * uy, length * uz);
	}
	updateCollision();
}

void xCheckCollisionDialog::moveMinusNormal()
{
	if (!checkTreeItemHasChild(CollisionParticle->currentItem())) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("클러스터를 선택해 주세요."));
		msg.exec();
		return;
	}
	xClusterInformation info;
	foreach(info, cinfos) {
		if (selectedCluster >= info.scid && selectedCluster < info.count / info.neach)
			break;
	}
	double length = LE_MoveLength->text().toDouble();
	QStringList normal = LE_Direction->text().split(",");
	double ux = -normal.at(0).toDouble();
	double uy = -normal.at(1).toDouble();
	double uz = -normal.at(2).toDouble();
	for (int i = 0; i < info.neach; i++) {
		xp->MoveParticle(info.sid + selectedCluster * info.neach + i, length * ux, length * uy, length * uz);
	}
	updateCollision();
}

//void xCheckCollisionDialog::changePosition(QTableWidgetItem * item)
//{
	/*if (!isSetup)
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
	}*/
//}

bool xCheckCollisionDialog::checkTreeItemHasChild(QTreeWidgetItem* item)
{
	return item->childCount();
}

void xCheckCollisionDialog::updateCollision()
{
	QTreeWidgetItem* item = CollisionParticle->currentItem();
	//QString itext = item->text(0);
	while (item->childCount()) {
		item = item->child(0);
		//itext = item->text(0);
	}
	item = item->parent();
	for (unsigned int i = 0; i < item->childCount(); i++) {
		QTreeWidgetItem* pair_item = item->child(i);
		//itext = pair_item->text(0);
		unsigned int pos = pair_item->text(0).indexOf(" ", 0);
		unsigned int id = pair_item->text(0).mid(1, pos - 1).toUInt();
		pos = pair_item->text(0).lastIndexOf("P");
		unsigned int jd = pair_item->text(0).mid(pos + 1).toUInt();
		float distance = xp->DistanceTwoParticlesFromSurface(id, jd);
		if (distance <= 0) {
			QTreeWidgetItem* parent = pair_item;
			while (parent->parent()) {
				parent = parent->parent();
			}
			QString ptext = parent->text(0);
			if (parent->childCount() >= 2) {
				if (pair_item->parent()->childCount() >= 2)
					delete pair_item;
				else
					delete pair_item->parent();
				//delete deleteItem;
			}
			else {
				delete parent;
				return;
			}
		}
		else{
			collision_info[pair_item->text(0)].overlap = distance;
			vector3f u = xp->NormalTwoParticles(id, jd);
			collision_info[pair_item->text(0)].ux = u.x;
			collision_info[pair_item->text(0)].uy = u.y;
			collision_info[pair_item->text(0)].uz = u.z;
		}
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
	//if (changedColor.size()) {
	//	QMapIterator<unsigned int, QColor> color(changedColor);
	//	QColor previous_color;
	//	while (color.hasNext()) {
	//		color.next();
	//		xp->ChangeColor(color.key(), color.value(), previous_color);
	//	}
	//	changedColor.clear();
	//}
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

void xCheckCollisionDialog::clickCancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
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
	
	QString prefix = item->text(0).mid(0, item->text(0).lastIndexOf(" "));
	if (prefix == "Cluster") {
		if (!item->parent()) {
			unsigned int id = item->text(0).mid(item->text(0).lastIndexOf(" ")).toUInt();
			//QColor previous_color;
			highlightSelectedCluster(id, 0);
			selectedCluster = id;
			//xp->ChangeColor(id, QColor(255, 0, 0), previous_color);
			//changedColor[id] = previous_color;
			//for (unsigned int i = 0; i < item->childCount(); i++) {
			//	QTreeWidgetItem* child = item->child(i);
			//	unsigned int jd = child->text(0).mid(child->text(0).lastIndexOf(" ")).toUInt();
			//	highlightSelectedCluster(jd, 0);
			//	/*xp->ChangeColor(jd, QColor(0, 255, 0), previous_color);
			//	changedColor[jd] = previous_color;*/
			//}
		}
		else if (item->parent() && item->childCount()) {
			unsigned int id = item->text(0).mid(item->text(0).lastIndexOf(" ")).toUInt();
			highlightSelectedCluster(id, 0);
			selectedCluster = id;
			//QColor previous_color;
			//xp->ChangeColor(id, QColor(255, 0, 0), previous_color);
			//changedColor[id] = previous_color;
			//QTreeWidgetItem* parent = item->parent();
			//unsigned int jd = parent->text(0).mid(parent->text(0).lastIndexOf(" ") + 1).toUInt();
			//highlightSelectedCluster(jd, 0);
			/*xp->ChangeColor(jd, QColor(0, 255, 0), previous_color);
			changedColor[jd] = previous_color;*/
		}		
	}
	else
	{
	    if (item->parent() && !item->childCount()) {
			LE_Overlap->clear();
			LE_Direction->clear();
			LE_MoveLength->clear();
			unsigned int pos = item->text(0).indexOf(" ", 0);
			unsigned int id = item->text(0).mid(1, pos - 1).toUInt();
			pos = item->text(0).lastIndexOf("P");
			unsigned int jd = item->text(0).mid(pos + 1).toUInt();
			QColor previous_color;
			xp->ChangeColor(id, QColor(255, 0, 0), previous_color);
			changedColor[id] = previous_color;
			xp->ChangeColor(jd, QColor(0, 255, 0), previous_color);
			changedColor[jd] = previous_color;
			if (collision_info.find(item->text(0)) != collision_info.end()) {
				xCollisionPair pair = collision_info[item->text(0)];
				LE_Overlap->setText(QString("%1").arg(pair.overlap));
				QString normal;
				QTextStream stream(&normal);
				stream.setRealNumberPrecision(4);
				stream << pair.ux << ", " << pair.uy << ", " << pair.uz;
				LE_Direction->setText(normal);
				LE_MoveLength->setText(QString("%1").arg((1.0 + 1e-6) * pair.overlap));
			}
			selectedCluster = -1;
		}
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
			cinfo.scid = ncp;
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
	//QStringList labels = { "X", "Y", "Z" };
	//Information->setColumnCount(3);
	//Information->setHorizontalHeaderLabels(labels);
	//if (cpos) {
	//	Information->setRowCount(ncp);
	//	for (unsigned int i = 0; i < ncp; i++) {
	//		QTableWidgetItem *item = nullptr;
	//		Information->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 0])));
	//		Information->setItem(i, 1, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 1])));
	//		Information->setItem(i, 2, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 2])));
	//		//Information->setItem(i, 3, new QTableWidgetItem(QString("%1").arg(cpos[i * 4 + 3])));
	//	}
	//}
	if(pos) delete[] pos;
	if(ep) delete[] ep;
	if(cpos) delete[] cpos;
	isSetup = true;
}

void xCheckCollisionDialog::checkCollision()
{
	collision_info.clear();  
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

	if (c_pairs.size()) {
		std::map<pair<unsigned int, unsigned int>, xPairData>::iterator it = c_pairs.begin();
		QString prefix = "Cluster ";
		for (; it != c_pairs.end(); it++) {
			unsigned int id = it->first.first;
			unsigned int jd = it->first.second;
			
			xClusterInformation info;
			for (unsigned int i = 0; i < count; i++) {
				if (id >= _cinfos[i].sid && id * _cinfos[i].neach < _cinfos[i].sid + _cinfos[i].count) {
					info = _cinfos[i];
					break;
				}					
			}
			xClusterInformation jnfo;
			for (unsigned int i = 0; i < count; i++) {
				if (jd >= _cinfos[i].sid && jd * _cinfos[i].neach < _cinfos[i].sid + _cinfos[i].count) {
					jnfo = _cinfos[i];
					break;
				}
			}
			unsigned int cid = (info.sid + id) / info.neach;
			QString si = prefix + QString("%1").arg(info.scid + cid);			
			if (iclusters.find(si) == iclusters.end()) {
				iclusters[si] = new QTreeWidgetItem(CollisionParticle);
				iclusters[si]->setText(0, si);
			}
			unsigned int cjd = (jnfo.sid + jd) / info.neach;
			QString sj = prefix + QString("%1").arg(jnfo.scid + cjd);
			if (jclusters.find(sj) == jclusters.end()) {
				jclusters[sj] = new QTreeWidgetItem(iclusters[si]);
				jclusters[sj]->setText(0, sj);
			}		
			QString cpair = "P" + QString("%1").arg(id) + " - " + "P" + QString("%1").arg(jd);
			QTreeWidgetItem* pair_item = new QTreeWidgetItem(jclusters[sj]);
			pair_item->setText(0, cpair);
			collision_info[cpair] =
			{
				it->second.gab,
				it->second.nx,
				it->second.ny,
				it->second.nz
			};
		}
	}
	else {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("충돌이 일어난 클러스터가 존재하지 않습니다."));
		msg.exec();
	}
	if (_cinfos) delete[] _cinfos;
	if (pos) delete[] pos;
	if (ep) delete[] ep;
	if (cpos) delete[] cpos;
}