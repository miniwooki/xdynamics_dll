#include "xparticle_result_dlg.h"
#include "xdynamics_manager/xResultManager.h"
#include "xdynamics_manager/xModel.h"
#include <QFileDialog>
#include <QTextStream>
#include "xMessageBox.h"

xparticle_result_dlg::xparticle_result_dlg(QWidget* parent)
	: QDialog(parent)
	, xrm(NULL)
	, m_part(0)
	, m_particle(0)
{
	setupUi(this);
	connect(PB_LoadTable, SIGNAL(clicked()), this, SLOT(click_load()));
	connect(PB_Export, SIGNAL(clicked()), this, SLOT(click_export()));
	connect(PB_Exit, SIGNAL(clicked()), this, SLOT(click_exit()));
	connect(LE_Part, SIGNAL(editingFinished()), this, SLOT(edit_target_part()));
	connect(LE_Time, SIGNAL(editingFinished()), this, SLOT(edit_target_particle()));
	connect(RB_Part, SIGNAL(clicked()), this, SLOT(enable_part()));
	connect(RB_Time, SIGNAL(clicked()), this, SLOT(enable_time()));
	//QStringList labels;
	//TW->setco
	part_labels << "id" << "PX" << "PY" << "PZ" << "VX" << "VY" << "VZ";
	time_labels << "time" << "PX" << "PY" << "PZ" << "VX" << "VY" << "VZ";
	//TW->setHorizontalHeaderLabels(labels);
	TW->verticalHeader()->hide();
	TW->setShowGrid(true);
}

xparticle_result_dlg::~xparticle_result_dlg()
{

}

void xparticle_result_dlg::edit_target_part()
{
	m_part = LE_Part->text().toUInt();
}

void xparticle_result_dlg::edit_target_particle()
{
	m_particle = LE_Time->text().toUInt();
}

void xparticle_result_dlg::enable_part()
{
	TW->clear();
	TW->setRowCount(0);
	TW->setColumnCount(0);
	LE_Time->setDisabled(true);
	LE_Part->setEnabled(true);
}

void xparticle_result_dlg::enable_time()
{
	TW->clear();
	TW->setRowCount(0);
	TW->setColumnCount(0);
	LE_Part->setDisabled(true);
	LE_Time->setEnabled(true);
}

void xparticle_result_dlg::setup(xResultManager * _xrm)
{
	xrm = _xrm;
	LE_Part->setText("0");
	LE_Time->setText("0");
	unsigned int npart = xrm->get_current_part_number();
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
		TW->setColumnCount(7);
		TW->setHorizontalHeaderLabels(part_labels);
		unsigned int pid = LE_Part->text().toUInt();
		unsigned int spid = pid * ns * 4;
		unsigned int svid = pid * ns * 3;
		for (unsigned int i = 0; i < ns; i++)
		{
			unsigned int s = spid + i*4;
			unsigned int v = svid + i*3;
			append(i, pos[s + 0], pos[s + 1], pos[s + 2], vel[v + 0], vel[v + 1], vel[v + 2]);
		}
	}
	else if (ltg == PARTICLE_BASE)
	{
		TW->setColumnCount(8);
		TW->setHorizontalHeaderLabels(part_labels);
		unsigned int npt = xrm->get_current_part_number();
		unsigned int tpt = LE_Time->text().toUInt();
		double* time = xrm->get_times();
	/*	std::function<void(int&)> spin = [](int &iteration) {
			con
		};*/
		for (unsigned int i = 0; i < npt; i++)
		{
			unsigned int s = tpt * 4 + i * ns * 4;
			unsigned int v = tpt * 3 + i * ns * 3;
			append_particle_base(i, time[i], pos[s + 0], pos[s + 1], pos[s + 2], vel[v + 0], vel[v + 1], vel[v + 2]);
		}
	}
}

void xparticle_result_dlg::click_export()
{
	if (!TW->rowCount())
	{
		xMessageBox::run("It must first be loaded to export the data.");
		return;
	}
	QString m_file;
	if (RB_Part->isChecked())
		m_file = QString("result_particles_of_part%1").arg(m_part);
	if (RB_Time->isChecked())
		m_file = QString("result_of_particle%1").arg(m_particle);
	QString dir = QString::fromStdString(xModel::makeFilePath(m_file.toStdString() + ".txt"));
	QString fileName = QFileDialog::getSaveFileName(this, tr("Export"), dir, tr("Text (*.txt)"));
	if (!fileName.isEmpty())
	{
		QFile qf(fileName);
		qf.open(QIODevice::WriteOnly);
		QTextStream qts(&qf);
		bool ck[7] = { CBPX->isChecked(), CBPY->isChecked(), CBPZ->isChecked(), CBVX->isChecked(), CBVY->isChecked(), CBVZ->isChecked() };
		if (RB_Part->isChecked())
		{
			qts.setFieldAlignment(QTextStream::FieldAlignment::AlignCenter);
			//qts.setFieldWidth(10);
			qts << part_labels.at(0)                << "         "
				<< (ck[0] ? part_labels.at(1) : "") << "         "
				<< (ck[1] ? part_labels.at(2) : "") << "         "
				<< (ck[2] ? part_labels.at(3) : "") << "         "
				<< (ck[3] ? part_labels.at(4) : "") << "         "
				<< (ck[4] ? part_labels.at(5) : "") << "         "
				<< (ck[5] ? part_labels.at(6) : "") << endl;
			for (unsigned int i = 0; i < TW->rowCount(); i++)
			{
				qts << QString("%1").arg(TW->item(i, 0)->text().toUInt(), -20);
				for (unsigned int j = 0; j < 6; j++)
				{
					qts << (ck[j] ? QString("%1").arg(TW->item(i, j + 1)->text().toDouble(), -20, 'f', 10) : "");
				}
				qts << endl;
			}
		}
		else if (RB_Time->isChecked())
		{
			qts << time_labels.at(0)			    << "         " 
				<< (ck[0] ? time_labels.at(1) : "") << "         "
				<< (ck[1] ? time_labels.at(2) : "") << "         "
				<< (ck[2] ? time_labels.at(3) : "") << "         "
				<< (ck[3] ? time_labels.at(4) : "") << "         "
				<< (ck[4] ? time_labels.at(5) : "") << "         "
				<< (ck[5] ? time_labels.at(6) : "") << "         "
				<< (ck[6] ? time_labels.at(7) : "") << endl;
			for (unsigned int i = 0; i < TW->rowCount(); i++)
			{
				qts << QString("%1").arg(TW->item(i, 0)->text().toDouble(), -20, 'f', 10);
				for (unsigned int j = 0; j < 7; j++)
				{
					qts << (ck[j] ? QString("%1").arg(TW->item(i, j + 1)->text().toDouble(), -20, 'f', 10) : "");
				}
				qts << endl;
			}
		}
		qf.close();
	}
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

void xparticle_result_dlg::append_particle_base(unsigned int i, double t, double px, double py, double pz, double vx, double vy, double vz)
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