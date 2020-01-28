#include "gen_cluster_dlg.h"
#include <QtWidgets>
#include <QtDataVisualization/qvalue3daxis.h>
#include <QtDataVisualization/q3dscene.h>
#include <QtDataVisualization/q3dcamera.h>
#include <QtDataVisualization/qscatter3dseries.h>
#include <QtDataVisualization/q3dtheme.h>
#include <QtDataVisualization/QCustom3DItem>


ScatterDataModifier::ScatterDataModifier(Q3DScatter *scatter)
	: m_graph(scatter)
	//, m_particle(nullptr)
{
	m_graph->setShadowQuality(QAbstract3DGraph::ShadowQualityNone);
	m_graph->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);
	minAxis = QVector3D(FLT_MAX, FLT_MAX, FLT_MAX);
	maxAxis = QVector3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
}

ScatterDataModifier::~ScatterDataModifier()
{
	if(m_particles.size()) qDeleteAll(m_particles);
}

void ScatterDataModifier::addParticle(int i, double x, double y, double z, double r)
{
	QCustom3DItem* pt = new QCustom3DItem;
	pt->setScaling(QVector3D(0.15,0.15,0.15));
	pt->setMeshFile(QStringLiteral(":/Resources/mesh/largesphere.obj"));
	QImage color = QImage(2, 2, QImage::Format_RGB32);
	color.fill(QColor(0xff, 0xbb, 0x00));
	pt->setTextureImage(color);
	m_graph->addCustomItem(pt);
	m_particles[i] = pt;
	if (minAxis.x() > x - r) minAxis.setX(x - r);
	if (minAxis.y() > y - r) minAxis.setY(y - r);
	if (minAxis.z() > z - r) minAxis.setZ(z - r);
	if (maxAxis.x() < x + r) maxAxis.setX(x + r);
	if (maxAxis.y() < y + r) maxAxis.setY(y + r);
	if (maxAxis.z() < z + r) maxAxis.setZ(z + r);
	m_graph->axisX()->setRange(minAxis.x(), maxAxis.x());
	m_graph->axisY()->setRange(minAxis.y(), maxAxis.y());
	m_graph->axisZ()->setRange(minAxis.z(), maxAxis.z());
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
	modifier = new ScatterDataModifier(graph);
	connect(SB_Rows, SIGNAL(valueChanged(int)), this, SLOT(increaseRows(int)));
}

gen_cluster_dlg::~gen_cluster_dlg()
{
	if (tableItems.size()) qDeleteAll(tableItems);
	if (graph) delete graph; graph = nullptr;
	//if (modifier) delete modifier; modifier = nullptr;
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
		modifier->addParticle(nrow, 0, 0, 0, 0.02);
	}
	
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
