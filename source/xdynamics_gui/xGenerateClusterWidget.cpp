#include "xGenerateClusterWidget.h"
#include "gen_cluster_dlg.h"

wgenclusters::wgenclusters(QWidget* parent)
	: QWidget(parent)
	, graph(nullptr)
{
	setupUi(this);
	connect(PB_Gen, &QPushButton::clicked, this, &wgenclusters::generateClusterParticles);
}

wgenclusters::~wgenclusters()
{
	if (graph) delete graph; graph = nullptr;
	if (modifier) delete modifier; modifier = nullptr;
}

void wgenclusters::setClusterView(unsigned int n, double* c)
{
	graph = new Q3DScatter;
	QWidget* container = QWidget::createWindowContainer(graph);
	QHBoxLayout *hlayout = new QHBoxLayout(ViewFrame);
	hlayout->addWidget(container, 1);
	hlayout->setMargin(0);
	graph->setAspectRatio(1.0);
	graph->axisX()->setSegmentCount(10);
	graph->axisY()->setSegmentCount(10);
	graph->axisZ()->setSegmentCount(10);
	modifier = new ScatterDataModifier(graph);
	connect(SBScale, SIGNAL(valueChanged(int)), this, SLOT(changeScale(int)));
	for (unsigned int i = 0; i < n; i++) {
		unsigned int id = i * 4;
		modifier->addParticle(i, c[id + 0], c[id + 1], c[id + 2], c[id + 3], SBScale->value());
	}
	
}

void wgenclusters::generateClusterParticles()
{
	QString name = LEName->text();
	int dim[3] =
	{ DX->text().toInt(), DY->text().toInt(), DZ->text().toInt() };
	double loc[3] =
	{ LEX->text().toDouble(), LEY->text().toDouble(), LEZ->text().toDouble() };
	emit clickedGenerateButton(name, dim, loc);
}

void wgenclusters::changeScale(int scale)
{
	modifier->setScale(scale);
}