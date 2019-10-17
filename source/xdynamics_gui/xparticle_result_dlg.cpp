#include "xparticle_result_dlg.h"
#include "xdynamics_manager/xResultManager.h"

xparticle_result_dlg::xparticle_result_dlg(QWidget* parent)
	: QDialog(parent)
	, xrm(NULL)
{
	setupUi(this);
	connect(PB_LoadTable, SIGNAL(clicked()), this, SLOT(click_load()));
	connect(PB_Export, SIGNAL(clicked()), this, SLOT(click_export()));
	connect(PB_Exit, SIGNAL(clicked()), this, SLOT(click_exit()));
	connect(LE_Part, SIGNAL())
	QStringList labels;
	labels << "id" << "PX" << "PY" << "PZ" << "VX" << "VY" << "VZ";
	TW->setHorizontalHeaderLabels(labels);
	TW->verticalHeader()->hide();
	TW->setShowGrid(true);
}

xparticle_result_dlg::~xparticle_result_dlg()
{

}

void xparticle_result_dlg::setup(xResultManager * _xrm)
{
	xrm = _xrm;
	unsigned int npart = xrm->get_num_parts();
	L_Part->setText(QString("/ %1").arg(npart));
	unsigned int nparticle = xrm->get_num_clusters();
	L_Time->setText(QString("/ %1").arg(nparticle));
}

void xparticle_result_dlg::click_load()
{
	unsigned int ns = xrm->get_num_clusters();
	unsigned int np = xrm->get_num_particles();
	float *pos = NULL;
	float *vel = NULL;
	double *acc = NULL;
	pos = xrm->get_particle_position_result_ptr();
	vel = xrm->get_particle_velocity_result_ptr();
	if (RB_Part->isChecked())
		ltg = PART_BASE;
	else
		ltg = PARTICLE_BASE;
	
	if (ltg == PART_BASE)
	{
		unsigned int pid = LE_Part->text().toUInt();
		unsigned int sid = pid * ns * 4;
		for (unsigned int i = 0; i < ns; i++)
		{
			unsigned int s = sid + i;
			append(i, pos[s + 0], pos[s + 1], pos[s + 2], vel[s + 0], vel[s + 1], vel[s + 2]);
		}
	}
	else if (ltg == PARTICLE_BASE)
	{
		unsigned int npt = xrm->get_num_parts();
		unsigned int tpt = LE_Time->text().toUInt();
		double* time = xrm->get_times();
		for (unsigned int i = 0; i < npt; i++)
		{
			unsigned int s = i * ns * 4 + tpt * 4;
			append_particle_base(time[i], pos[s + 0], pos[s + 1], pos[s + 2], vel[s + 0], vel[s + 1], vel[s + 2]);
		}
	}
}

void xparticle_result_dlg::click_export()
{

}

void xparticle_result_dlg::click_exit()
{
	this->close();
	this->setResult(QDialog::Rejected);
}

void xparticle_result_dlg::append(unsigned int id, double px, double py, double pz, double vx, double vy, double vz)
{
	unsigned int row = TW->rowCount();
	QTableWidgetItem *ID = new QTableWidgetItem(QString("%1").arg(id));
	QTableWidgetItem *PX = new QTableWidgetItem(QString("%1").arg(px));
	QTableWidgetItem *PY = new QTableWidgetItem(QString("%1").arg(py));
	QTableWidgetItem *PZ = new QTableWidgetItem(QString("%1").arg(pz));
	QTableWidgetItem *VX = new QTableWidgetItem(QString("%1").arg(vx));
	QTableWidgetItem *VY = new QTableWidgetItem(QString("%1").arg(vy));
	QTableWidgetItem *VZ = new QTableWidgetItem(QString("%1").arg(vz));
	int col = 0;
	TW->insertRow(row);
	TW->setItem(row, col++, ID);
	TW->setItem(row, col++, PX);
	TW->setItem(row, col++, PY);
	TW->setItem(row, col++, PZ);
	TW->setItem(row, col++, VX);
	TW->setItem(row, col++, VY);
	TW->setItem(row, col++, VZ);
}

void xparticle_result_dlg::append_particle_base(double t, double px, double py, double pz, double vx, double vy, double vz)
{
	unsigned int row = TW->rowCount();
	QTableWidgetItem *T = new QTableWidgetItem(QString("%1").arg(t));
	QTableWidgetItem *PX = new QTableWidgetItem(QString("%1").arg(px));
	QTableWidgetItem *PY = new QTableWidgetItem(QString("%1").arg(py));
	QTableWidgetItem *PZ = new QTableWidgetItem(QString("%1").arg(pz));
	QTableWidgetItem *VX = new QTableWidgetItem(QString("%1").arg(vx));
	QTableWidgetItem *VY = new QTableWidgetItem(QString("%1").arg(vy));
	QTableWidgetItem *VZ = new QTableWidgetItem(QString("%1").arg(vz));
	int col = 0;
	TW->insertRow(row);
	TW->setItem(row, col++, T);
	TW->setItem(row, col++, PX);
	TW->setItem(row, col++, PY);
	TW->setItem(row, col++, PZ);
	TW->setItem(row, col++, VX);
	TW->setItem(row, col++, VY);
	TW->setItem(row, col++, VZ);
}