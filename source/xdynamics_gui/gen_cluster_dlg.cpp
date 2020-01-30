#include "gen_cluster_dlg.h"
#include <QtWidgets>
#include <QtDataVisualization/qvalue3daxis.h>
#include <QtDataVisualization/q3dscene.h>
#include <QtDataVisualization/q3dcamera.h>
#include <QtDataVisualization/qscatter3dseries.h>
#include <QtDataVisualization/q3dtheme.h>
#include <QtDataVisualization/QCustom3DItem>

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
	pt->setScaling(QVector3D(s,s,s));
	pt->setMeshFile(QStringLiteral(":/Resources/mesh/largesphere.obj"));
	QImage color = QImage(2, 2, QImage::Format_RGB32);
	color.fill(QColor(0xff, 0xbb, 0x00));
	pt->setTextureImage(color);
	m_graph->addCustomItem(pt);
	m_particles[i] = pt;
	//if (minAxis.x() > x - r) minAxis.setX(x - r);
	//if (minAxis.y() > y - r) minAxis.setY(y - r);
	//if (minAxis.z() > z - r) minAxis.setZ(z - r);
	//if (maxAxis.x() < x + r) maxAxis.setX(x + r);
	//if (maxAxis.y() < y + r) maxAxis.setY(y + r);
	//if (maxAxis.z() < z + r) maxAxis.setZ(z + r);
	double div = 1.0 / scale;
	m_graph->axisX()->setRange(-1 * div, 1 * div);// minAxis.x(), maxAxis.x());
	m_graph->axisY()->setRange(-1 * div, 1 * div);// minAxis.y(), maxAxis.y());
	m_graph->axisZ()->setRange(-1 * div, 1 * div);// minAxis.z(), maxAxis.z());
	if (maxRadius < r)
		maxRadius = r;
	m_radius[i] = r;

}

gen_cluster_dlg::gen_cluster_dlg(QWidget* parent)
	: QDialog(parent)
	, modifier(nullptr)
{
	setupUi(this);
	SB_Rows->setValue(0);
	SB_Numbers->setValue(0);
	InputTable->setColumnCount(4);
	QStringList labels = { "X", "Y", "Z", "R" };
	InputTable->setHorizontalHeaderLabels(labels);
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
	connect(InputTable, &QTableWidget::cellClicked, this, &gen_cluster_dlg::clickCell);
	connect(InputTable, &QTableWidget::itemChanged, this, &gen_cluster_dlg::changeItem);
	rc[0] = -1;
	rc[1] = -1;
}

gen_cluster_dlg::~gen_cluster_dlg()
{
	if (tableItems.size()) qDeleteAll(tableItems);
	if (graph) delete graph; graph = nullptr;
	if (modifier) delete modifier; modifier = nullptr;
}

void gen_cluster_dlg::increaseRows(int nrow)
{
	int previous_rows = InputTable->rowCount();
	
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
		nrow--;
		QTableWidgetItem *x = new QTableWidgetItem("0");
		QTableWidgetItem *y = new QTableWidgetItem("0");
		QTableWidgetItem *z = new QTableWidgetItem("0");
		QTableWidgetItem *r = new QTableWidgetItem("0.02");
		tableItems[QPair<int, int>(nrow, 0)] = x;
		tableItems[QPair<int, int>(nrow, 1)] = y;
		tableItems[QPair<int, int>(nrow, 2)] = z;
		tableItems[QPair<int, int>(nrow, 3)] = r;
		InputTable->setItem(nrow, 0, x);
		InputTable->setItem(nrow, 1, y);
		InputTable->setItem(nrow, 2, z);
		InputTable->setItem(nrow, 3, r);
		modifier->addParticle(nrow, 0, 0, 0, 0.02, SB_Scale->value());
	}
	
}

void gen_cluster_dlg::changeScale(int scale)
{
	modifier->setScale(scale);
}

void gen_cluster_dlg::clickAdd()
{

}

void gen_cluster_dlg::clickApply()
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
	if (rc[0] < 0 && rc[1] < 0)
		return;
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

void gen_cluster_dlg::clickCell(int r, int c)
{
	rc[0] = r;
	rc[1] = c;
}
