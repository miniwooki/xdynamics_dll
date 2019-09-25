#include "xpass_distribution_dlg.h"
#include "xdynamics_object/xPlaneObject.h"
#include "xdynamics_manager/xResultManager.h"
#include "xdynamics_object/xParticlePlaneContact.h"
#include "xListWidget.h"

xpass_distribution_dlg::xpass_distribution_dlg(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);
	connect(PB_SelectPart, SIGNAL(clicked()), this, SLOT(click_select()));
	connect(PB_ANALYSIS, SIGNAL(clicked()), this, SLOT(click_analysis()));
	connect(PB_Exit, SIGNAL(clicked()), this, SLOT(click_exit()));
}

xpass_distribution_dlg::~xpass_distribution_dlg()
{

}

void xpass_distribution_dlg::setup(xResultManager * _xrm, QStringList qls)
{
	frlist = qls;
	xrm = _xrm;
	PROB->setValue(0);
}

void xpass_distribution_dlg::click_select()
{
	xListWidget lw;// = new xListWidget;
	//QStringList result_file_list;
	//xlist<xstring>* flist = xrm->get_part_file_list();
	
	//for (QList<QString>::iterator it = frlist.begin(); it != frlist.end(); it++)
	//{
	//	result_file_list.push_back();
	//}
	if (frlist.size())
	{
		lw.setup_widget(frlist);//lw->addItems(result_file_list);
		int ret = lw.exec();
		if (ret)
		{
			qslist = lw.get_selected_items();
		
		}
	}
}

void xpass_distribution_dlg::click_analysis()
{
	QStringList point = LEP0->text().split(",");
	vector3d p0 = new_vector3d(0, 0, 0); 
	vector3d p1 = new_vector3d(0, 0, 0); 
	vector3d p2 = new_vector3d(0, 0, 0);
	vector3d p3 = new_vector3d(0, 0, 0);
	if(point.size() == 3) 
		vector3d p0 = { point.at(0).toDouble(), point.at(1).toDouble(), point.at(2).toDouble() };
	point = LEP1->text().split(",");
	if (point.size() == 3) 
		vector3d p1 = { point.at(0).toDouble(), point.at(1).toDouble(), point.at(2).toDouble() };
	point = LEP2->text().split(",");
	if (point.size() == 3) 
		vector3d p2 = { point.at(0).toDouble(), point.at(1).toDouble(), point.at(2).toDouble() };
	point = LEP3->text().split(",");
	if (point.size() == 3) 
		vector3d p3 = { point.at(0).toDouble(), point.at(1).toDouble(), point.at(2).toDouble() };

	xPlaneObject plane("area");
	plane.define(p0, p1, p2, p3);
	xParticlePlaneContact c("passing");
	float* ptrs = xrm->get_particle_position_result_ptr();
	QString spart = qslist.at(0);
	int begin = spart.lastIndexOf("/");
	QString mm = spart.mid(begin + 5, 4);
	unsigned int sp = mm.toUInt();
	//float *ptrs = xrm->get_particle_position_result_ptr();
	unsigned int np = xrm->get_num_particles();
	/*bool* is_contact = new bool[np];
	memset(is_contact, 0, sizeof(bool)*np);*/
	QMap<unsigned int, unsigned int> id;
	for (unsigned int i = 0; i < np; i++) id[i]=i;
	PROB->setMaximum(qslist.size() * id.size());
	unsigned int cnt = 1;
	for (unsigned int i = 0; i < qslist.size(); i++)
	{
		unsigned int pnid = cid.size();
		vector4f* ps = (vector4f*)(ptrs + ((sp + i) * np * 4));
		foreach(unsigned int j, id)
		{
			bool b = c.detect_contact(ps[j], plane);
			if(b)
				cid.push_back(j);
			PROB->setValue(cnt++);
		}
		unsigned int anid = cid.size();
		for (unsigned int k = pnid; k < anid; k++)
		{
			unsigned int rid = cid.at(k);
			id.remove(rid);
		}
	}
}

QList<unsigned int>& xpass_distribution_dlg::get_distribution_result()
{
	return cid;
}

void xpass_distribution_dlg::click_exit()
{
	this->close();
	this->setResult(QDialog::Accepted);
}