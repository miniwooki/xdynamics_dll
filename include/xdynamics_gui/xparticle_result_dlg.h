#pragma once
#include "ui_wparticle_result.h"
#include <QStringList>

class xResultManager;

class xparticle_result_dlg : public QDialog, private Ui::ParticleResultDialog
{
	enum load_target{ PART_BASE = 0, PARTICLE_BASE };
	Q_OBJECT
public:
	xparticle_result_dlg(QWidget* parent = NULL);
	~xparticle_result_dlg();

	void setup(xResultManager* _xrm);
//QMap<unsigned int, distribution_data>& get_distribution_result();
private:
	void append(unsigned int id, double px, double py, double pz, double vx, double vy, double vz);
	void append_particle_base(unsigned int i, double t, double px, double py, double pz, double vx, double vy, double vz);

private slots:
	void click_load();
	void click_export();
	void click_exit();
	void edit_target_part();
	void edit_target_particle();
	void enable_part();
	void enable_time();

private:
	unsigned int m_part;
	unsigned int m_particle;
	xResultManager* xrm;
	load_target ltg;
	QStringList part_labels;
	QStringList time_labels;
};