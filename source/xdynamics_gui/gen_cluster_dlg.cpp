#include "gen_cluster_dlg.h"
#include <QtWidgets>
#include <QtDataVisualization/qvalue3daxis.h>
#include <QtDataVisualization/q3dscene.h>
#include <QtDataVisualization/q3dcamera.h>
#include <QtDataVisualization/qscatter3dseries.h>
#include <QtDataVisualization/q3dtheme.h>
#include <QtDataVisualization/QCustom3DItem>

#include "xdynamics_object/xClusterObject.h"

#define SCALE_FACTOR 0.115

ScatterDataModifier::ScatterDataModifier(Q3DScatter *scatter)
	: m_graph(scatter)
	//, m_particle(nullptr)
{
	m_graph->setShadowQuality(QAbstract3DGraph::ShadowQualityNone);
	m_graph->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);
	maxRadius = -FLT_MAX;
	m_graph->axisX()->setTitle("X");
	m_graph->axisX()->setTitle("Y");
	m_graph->axisX()->setTitle("Z");
	m_graph->activeTheme()->setLabelBackgroundEnabled(false);
	//minAxis = QVector3D(FLT_MAX, FLT_MAX, FLT_MAX);
	//maxAxis = QVector3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
}

ScatterDataModifier::~ScatterDataModifier()
{
	//if(m_particles.size()) qDeleteAll(m_particles);
	//if (m_graph) delete m_graph; m_graph = nullptr;
}

void ScatterDataModifier::setScale(double scale)
{
	if (!m_particles.size())
		return;
	int i = 0;
	foreach(QCustom3DItem* item, m_particles) {
		//double ratio = m_radius[i++] / maxRadius;
		double s = SCALE_FACTOR * scale * m_radius[i++];
		item->setScaling(QVector3D(s, s, s));
		//item->setPosition(scale * item->position());
	}
	double div = 1.0 / scale;
	m_graph->axisX()->setRange(-1 * div, 1 * div);// minAxis.x(), maxAxis.x());
	m_graph->axisY()->setRange(-1 * div, 1 * div);// minAxis.y(), maxAxis.y());
	m_graph->axisZ()->setRange(-1 * div, 1 * div);// minAxis.z(), maxAxis.z());
}

void ScatterDataModifier::setRadius(int i, double radius, double scale)
{
	if (!m_particles.size())
		return;
	if (m_particles.find(i) == m_particles.end())
		return;
	//double ratio = radius / maxRadius;
	double s = SCALE_FACTOR * scale * radius;
	m_particles[i]->setScaling(QVector3D(s, s, s));
	/*if (radius > maxRadius)
		maxRadius = radius;*/
	m_radius[i] = radius;
}

void ScatterDataModifier::removeLastParticle()
{
	int id = m_particles.size() - 1;
	QCustom3DItem* item = m_particles.take(id);
	m_graph->removeCustomItem(item);
	m_radius.take(id);
	//delete item;
}

void ScatterDataModifier::setPosition(int i, double x, double y, double z, double scale)
{
	if (!m_particles.size())
		return;
	if (m_particles.find(i) == m_particles.end())
		return;
	m_particles[i]->setPosition(QVector3D(x, y, z));
}

void ScatterDataModifier::addParticle(int i, double x, double y, double z, double r, double scale)
{
	QCustom3DItem* pt = new QCustom3DItem;
	double s = SCALE_FACTOR * r * scale;
	pt->setPosition(QVector3D(x, y, z));
	pt->setScaling(QVector3D(s,s,s));
	pt->setMeshFile(QStringLiteral(":/Resources/mesh/largesphere.obj"));
	QImage color = QImage(2, 2, QImage::Format_RGB32);
	color.fill(QColor(0xff, 0xbb, 0x00));
	pt->setTextureImage(color);
	m_graph->addCustomItem(pt);
	m_particles[i] = pt;
	double div = 1.0 / scale;
	m_graph->axisX()->setRange(-1 * div, 1 * div);// minAxis.x(), maxAxis.x());
	m_graph->axisY()->setRange(-1 * div, 1 * div);// minAxis.y(), maxAxis.y());
	m_graph->axisZ()->setRange(-1 * div, 1 * div);// minAxis.z(), maxAxis.z());
	if (maxRadius < r)
		maxRadius = r;
	m_radius[i] = r;

}

void ScatterDataModifier::reset()
{
	m_graph->removeCustomItems();
	//qDeleteAll(m_particles);
}

unsigned int ScatterDataModifier::particleCount()
{
	return m_particles.size();
}

gen_cluster_dlg::gen_cluster_dlg(QWidget* parent)
	: QDialog(parent)
	, modifier(nullptr)
	//, isExistInList(false)
	, isChangeCluster(false)
	, isNewCluster(false)
	, isClickedCell(false)
	, current_list(nullptr)
{
	setupUi(this);
	SB_Rows->setValue(0);
	SB_Numbers->setValue(0);
	InputTable->setColumnCount(4);
	SB_Scale->setValue(1);
	graph = new Q3DScatter;
	QWidget *container = QWidget::createWindowContainer(graph);
	QHBoxLayout *hLayout = new QHBoxLayout(View3D);
	hLayout->addWidget(container, 1);

	//View3D->createWindowContainer(graph);
	if (!graph->hasContext()) {
		QMessageBox msgBox;
		msgBox.setText("Couldn't initialize the OpenGL context.");
		msgBox.exec();
		this->clickCancel();
	}
	//qreal aratio = graph->aspectRatio();
	graph->setAspectRatio(1.0);
	graph->axisX()->setSegmentCount(10);
	graph->axisY()->setSegmentCount(10);
	graph->axisZ()->setSegmentCount(10);
	modifier = new ScatterDataModifier(graph);
	connect(SB_Rows, SIGNAL(valueChanged(int)), this, SLOT(increaseRows(int)));
	connect(SB_Scale, SIGNAL(valueChanged(int)), this, SLOT(changeScale(int)));
	connect(ClusterList, &QListWidget::itemClicked, this, &gen_cluster_dlg::clickClusterItem);
	connect(InputTable, &QTableWidget::cellClicked, this, &gen_cluster_dlg::clickCell);
	connect(PB_New, &QPushButton::clicked, this, &gen_cluster_dlg::clickNew);
	connect(PB_Add, &QPushButton::clicked, this, &gen_cluster_dlg::clickAdd);
	connect(PB_Gen, &QPushButton::clicked, this, &gen_cluster_dlg::clickGen);
	connect(InputTable, &QTableWidget::itemChanged, this, &gen_cluster_dlg::changeItem);
	connect(InputTable, &QTableWidget::cellActivated, this, &gen_cluster_dlg::changeCell);
	rc[0] = -1;
	rc[1] = -1;
}

gen_cluster_dlg::~gen_cluster_dlg()
{
	if (tableItems.size()) qDeleteAll(tableItems);
	if (graph) delete graph; graph = nullptr;
	if (modifier) delete modifier; modifier = nullptr;
	QMapIterator<QString, QPair<QString, xClusterObject*>> cluster(clusters);
	while (cluster.hasNext()) {
		cluster.next();
		delete cluster.value().second;
	}
	clusters.clear();
	QMapIterator<QString, MapTableItems> table(tables);
	while (table.hasNext()) {
		table.next();
		qDeleteAll(table.value());
	}
	/*foreach(MapTableItems item, tables) {
		qDeleteAll
	}*/
}

void gen_cluster_dlg::prepareShow()
{
	InputTable->clear();
	modifier->reset();
}

void gen_cluster_dlg::deleteCurrentTableItems()
{
	for (unsigned int r = 0; r < InputTable->rowCount(); r++) {
		QTableWidgetItem* item = nullptr;
		item = InputTable->item(r, 0); delete item;
		item = InputTable->item(r, 1); delete item;
		item = InputTable->item(r, 2); delete item;
		item = InputTable->item(r, 3); delete item;
	}
	InputTable->clear();
}

void gen_cluster_dlg::checkNeedAdd()
{
	if (isNewCluster && InputTable->rowCount() > 1) {
		QMessageBox msg;
		msg.setText(
			QString::fromLocal8Bit("리스트에 추가되지 않은 현재 클러스터를 추가하시겠습니까?"));
		msg.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
		msg.setDefaultButton(QMessageBox::Ok);
		int ret = msg.exec();
		if (ret == QMessageBox::Ok) {
			clickAdd();
		}
	}
	else if (isChangeCluster) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("현재 클러스터에 변경된 내용이 있습니다.\n갱신하시겠습니까?"));
		msg.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
		msg.setDefaultButton(QMessageBox::Ok);
		int ret = msg.exec();
		if (ret == QMessageBox::Ok) {
			clickUpdate();
		}

	}
}

void gen_cluster_dlg::clickUpdate()
{
	MapTableItems items = tables.take(current_list->text());
	QString name = current_list->text();
	//qDeleteAll(items);
	QListWidgetItem* item = ClusterList->takeItem(ClusterList->currentRow());
	delete item;
	QPair<QString, xClusterObject*> cluster = clusters.take(name);
	delete cluster.second;
	clickAdd();
	isChangeCluster = false;
}

void gen_cluster_dlg::clickNew()
{
	checkNeedAdd();
	rc[0] = -1;
	rc[1] = -1;
	isNewCluster = true;
	isChangeCluster = false;
	current_list = nullptr;
	QStringList labels = { "X", "Y", "Z", "R" };
	InputTable->setHorizontalHeaderLabels(labels);
	LE_Name->setText(QString("Cluster%1").arg(clusters.size() + 1));
	SB_Rows->setValue(1);
	
	SB_Numbers->setValue(1);
	PB_Add->setText("Add");
	PB_Add->disconnect();
	connect(PB_Add, &QPushButton::clicked, this, &gen_cluster_dlg::clickAdd);
	isClickedCell = false;
}

void gen_cluster_dlg::clickClusterItem(QListWidgetItem* item)
{
	checkNeedAdd();
	modifier->reset();
	//clickNew();
	QString name = item->text();
	if (clusters.find(name) == clusters.end())
		return;
	if (tables.find(name) == tables.end())
		return;
	isChangeCluster = false;
	isNewCluster = false;
	QPair<QString, xClusterObject*> pair = clusters[name];
	Information->setText(pair.first);
	loadExistLocalPosition(name);
	current_list = item;
	SB_Rows->setValue(pair.second->NumElement());
	SB_Numbers->setValue(pair.second->TotalClusters());
	LE_Name->setText(name);
	PB_Add->setText("Update");
	PB_Add->disconnect();
	connect(PB_Add, &QPushButton::clicked, this, &gen_cluster_dlg::clickUpdate);
}

void gen_cluster_dlg::loadExistLocalPosition(QString name)
{
	InputTable->clear();
	if (isNewCluster)
		deleteCurrentTableItems();
	if (tables.find(name) == tables.end())
		return;
	rc[0] = -1;
	rc[1] = -1;
	QStringList labels = { "X", "Y", "Z", "R" };
	InputTable->setHorizontalHeaderLabels(labels);
	MapTableItems items = tables[name];
	//QMapIterator<QPair<int, int>, QTableWidgetItem*> iter(items);
	InputTable->setRowCount(items.size() / 4);
	for (int i = 0; i < InputTable->rowCount(); i++) {
		QTableWidgetItem* titems[4] = {
			items[QPair<int, int>(i, 0)],
			items[QPair<int, int>(i, 1)],
			items[QPair<int, int>(i, 2)],
			items[QPair<int, int>(i, 3)]
		};
		setRowData(i, titems);
	}
	isNewCluster = false;
	isClickedCell = false;
}

void gen_cluster_dlg::setRowData(int i, QTableWidgetItem** items)
{
	if (i >= InputTable->rowCount())
		return;
	/*QTableWidgetItem *x = new QTableWidgetItem(QString("%1").arg(_x));
	QTableWidgetItem *y = new QTableWidgetItem(QString("%1").arg(_y));
	QTableWidgetItem *z = new QTableWidgetItem(QString("%1").arg(_z));
	QTableWidgetItem *r = new QTableWidgetItem(QString("%1").arg(_r));*/
	tableItems[QPair<int, int>(i, 0)] = items[0];
	tableItems[QPair<int, int>(i, 1)] = items[1];
	tableItems[QPair<int, int>(i, 2)] = items[2];
	tableItems[QPair<int, int>(i, 3)] = items[3];
	InputTable->setItem(i, 0, items[0]);
	InputTable->setItem(i, 1, items[1]);
	InputTable->setItem(i, 2, items[2]);
	InputTable->setItem(i, 3, items[3]);
	modifier->addParticle(
		i, 
		items[0]->text().toDouble(), 
		items[1]->text().toDouble(),
		items[2]->text().toDouble(), 
		items[3]->text().toDouble(), 
		SB_Scale->value());
}

void gen_cluster_dlg::increaseRows(int nrow)
{
	if (!isNewCluster && !current_list)
		return;
	if (!isNewCluster)
		isChangeCluster = true;
	int previous_rows = InputTable->rowCount();
	isClickedCell = false;
	if (previous_rows > nrow && previous_rows > 0) {
		QTableWidgetItem* item = tableItems.take(QPair<int, int>(nrow, 0));
		delete item;
		item = tableItems.take(QPair<int, int>(nrow, 1));
		delete item;
		item = tableItems.take(QPair<int, int>(nrow, 2));
		delete item;
		item = tableItems.take(QPair<int, int>(nrow, 3));
		delete item;
		InputTable->setRowCount(nrow);
		modifier->removeLastParticle();
	}
	else if (previous_rows < nrow) {
		InputTable->setRowCount(nrow);
		for (unsigned int i = 0; i < nrow - previous_rows; i++) {			
			unsigned int nr = previous_rows + i;
			QTableWidgetItem *x = new QTableWidgetItem("0");
			QTableWidgetItem *y = new QTableWidgetItem("0");
			QTableWidgetItem *z = new QTableWidgetItem("0");
			QTableWidgetItem *r = new QTableWidgetItem("0.02");
			tableItems[QPair<int, int>(nr, 0)] = x;
			tableItems[QPair<int, int>(nr, 1)] = y;
			tableItems[QPair<int, int>(nr, 2)] = z;
			tableItems[QPair<int, int>(nr, 3)] = r;
			InputTable->setItem(nr, 0, x);
			InputTable->setItem(nr, 1, y);
			InputTable->setItem(nr, 2, z);
			InputTable->setItem(nr, 3, r);
			modifier->addParticle(nr, 0, 0, 0, 0.02, SB_Scale->value());
		}		
	}
	
}

void gen_cluster_dlg::changeScale(int scale)
{
	if (!isNewCluster && !current_list)
		return;
	modifier->setScale(scale);
	isClickedCell = false;
}

void gen_cluster_dlg::clickAdd()
{
	unsigned int num = SB_Numbers->value();
	if (!num)
	{
		QMessageBox msg;
		msg.setText("Input the number of clusters.");
		msg.exec();
		return;
	}
	if (InputTable->rowCount() == 0) {
		QMessageBox msg;
		msg.setText(QString::fromLocal8Bit("저장할 데이터가 존재하지 않습니다."));
		msg.exec();
		return;
	}
	QString name = LE_Name->text();
	xClusterObject* cObj = new xClusterObject(name.toStdString());
	vector4d* local = new vector4d[InputTable->rowCount()];
	double min_r = FLT_MAX;
	double max_r = -FLT_MAX;
	MapTableItems items;
	for (unsigned int i = 0; i < InputTable->rowCount(); i++) {
		double x = InputTable->item(i, 0)->text().toDouble();
		double y = InputTable->item(i, 1)->text().toDouble();
		double z = InputTable->item(i, 2)->text().toDouble();
		double r = InputTable->item(i, 3)->text().toDouble();
		local[i] = new_vector4d(x, y, z, r);
		if (min_r > r)
			min_r = r;
		if (max_r < r)
			max_r = r;
		items[QPair<int, int>(i, 0)] = InputTable->takeItem(i, 0);
		items[QPair<int, int>(i, 1)] = InputTable->takeItem(i, 1);
		items[QPair<int, int>(i, 2)] = InputTable->takeItem(i, 2);
		items[QPair<int, int>(i, 3)] = InputTable->takeItem(i, 3);
	}
	cObj->setClusterSet(InputTable->rowCount(), min_r, max_r, local, 0);
	cObj->SetTotalClusters(num);
	QString info;
	QTextStream stream(&info);
	//stream << qSetFieldWidth(12) << std::left;
	stream
		<< "Name"			<< " : " << name << endl
		<< "Count"			<< " : " << num << endl
		<< "Min. radius"	<< " : " << min_r << endl
		<< "Max. radius"	<< " : " << max_r << endl;
	QPair<QString, xClusterObject*> pair(info, cObj);
	clusters[name] = pair;
	QListWidgetItem* item = new QListWidgetItem(name);
	ClusterList->addItem(item);
	InputTable->setRowCount(0);
	tables[name] = items;
	tableItems.clear();
	modifier->reset();
	isNewCluster = false;
	isClickedCell = false;
	Information->clear();
	delete[] local;
}

void gen_cluster_dlg::clickGen()
{
	this->close();
	this->setResult(QDialog::Accepted);
}

void gen_cluster_dlg::clickCancel()
{
	this->close();
	this->setResult(QDialog::Rejected);
}

void gen_cluster_dlg::changeItem(QTableWidgetItem* item)
{
	
	/*qDebug() 
		<< "Current row : " << InputTable->currentRow()
		<< ", Current col : " << InputTable->currentColumn();*/
	if (!modifier->particleCount())
		return;
	if (rc[0] < 0 && rc[1] < 0)
		return;
	rc[0] = InputTable->currentRow();
	rc[1] = InputTable->currentColumn();
	if (rc[1] < 3) {
		switch (rc[1]) {
		case 0: modifier->setPosition(rc[0], item->text().toDouble(), InputTable->item(rc[0], 1)->text().toDouble(), InputTable->item(rc[0], 2)->text().toDouble(), SB_Scale->value()); break;
		case 1: modifier->setPosition(rc[0], InputTable->item(rc[0], 0)->text().toDouble(), item->text().toDouble(), InputTable->item(rc[0], 2)->text().toDouble(), SB_Scale->value()); break;
		case 2: modifier->setPosition(rc[0], InputTable->item(rc[0], 0)->text().toDouble(), InputTable->item(rc[0], 1)->text().toDouble(), item->text().toDouble(), SB_Scale->value()); break;
		}
	}
	else {
		double s = SB_Scale->value();
		double r = item->text().toDouble();
		modifier->setRadius(rc[0], r, s);
	}
}

void gen_cluster_dlg::changeCell(int r, int c)
{
	if (isClickedCell) {
		rc[0] = r;
		rc[1] = c;
	}
}

void gen_cluster_dlg::clickCell(int r, int c)
{
	rc[0] = r;
	rc[1] = c;
	isClickedCell = true;
}
