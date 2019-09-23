#include "xResultCallThread.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include "xvAnimationController.h"
#include <QTextStream>
#include <QFile>
#include <QDir>
//#include <sstream>
//#include <iostream>

xResultCallThread::xResultCallThread()
	: is_success_loading_model(false)
{

}

xResultCallThread::~xResultCallThread()
{

}

void xResultCallThread::set_dynamics_manager(xDynamicsManager * _xdm, QString fpath)
{
	xdm = _xdm;
	path = fpath;
	is_success_loading_model = xdm->upload_model_results(path.toStdString());
}

QStringList xResultCallThread::get_file_list()
{
	return flist;
}

void xResultCallThread::run()
{
	if (xdm && is_success_loading_model)
	{
		bool is_terminated = false;
		xResultManager* xrm = xdm->XResult();
		QString file_path;
		QTextStream ss(&file_path);
		//stringstream ss(file_path);
		unsigned int _nparts = 0;
		//	ss << path << "/part" << setw(4) << setfill('0') << _npart++ << ".bin";
			//list<std::string> file_list;
		while (1)
		{
			file_path.clear();
			file_path.sprintf("%spart%04d.bin", path.toStdString().c_str(), _nparts);
			//ss << path << "part" << setw(4) << setfill('0') << _nparts << ".bin";
			if (QDir().exists(file_path))
			{
				flist.push_back(file_path);
				_nparts++;
			}
			else
				break;
		}
		emit result_call_send_progress(5, "");
		if (xrm->get_num_parts() != _nparts)
		{
			emit result_call_send_progress(-1,
				QString("!_This result of this analysis is not enough[%1 / %2]. This may mean that the interpretation was interrupted.").arg(_nparts).arg(xrm->get_num_parts()));
			xrm->set_terminated_num_parts(_nparts);
			is_terminated = true;
		}			
		else
			emit result_call_send_progress(-1, QString("Detected part results are %1.").arg(_nparts));
	//	std::fstream fs;

		unsigned int cnt = 0;
		unsigned int np = xrm->get_num_particles();
		unsigned int ns = xrm->get_num_clusters();
		double ct = 0;
		double* _cpos = NULL, *_pos = NULL, *_vel = NULL, *_acc = NULL, *_ep = NULL, *_ev = NULL, *_ea = NULL;
		if (np)
		{
			_pos = new double[np * 4];
			_vel = new double[ns * 3];
			_acc = new double[ns * 3];
			_ep = new double[ns * 4];
			_ev = new double[ns * 4];
			_ea = new double[ns * 4];
			if (ns != np)
				_cpos = new double[ns * 4];
		}
		emit result_call_send_progress(10, "");
		foreach(QString it, flist)
		{
			QFile fs(it);
			fs.open(QIODevice::ReadOnly);
			//fs.open(it.value().toStdString(), std::ios::in | std::ios::binary);
			fs.read((char*)&ct, sizeof(double));
			xrm->get_times()[cnt] = ct;
			//time[cnt] = ct;
			if (np)
			{
				unsigned int _npt = 0;
				unsigned int _nct = 0;
				fs.read((char*)&_npt, sizeof(unsigned int));
				fs.read((char*)&_nct, sizeof(unsigned int));
				fs.read((char*)_pos, sizeof(double) * np * 4);
				fs.read((char*)_vel, sizeof(double) * ns * 3);
				fs.read((char*)_acc, sizeof(double) * ns * 3);
				fs.read((char*)_ep, sizeof(double) * ns * 4);
				fs.read((char*)_ev, sizeof(double) * ns * 4);
				fs.read((char*)_ea, sizeof(double) * ns * 4);
				if (ns != np)
					fs.read((char*)_cpos, sizeof(double) * ns * 4);
				xrm->save_dem_result(cnt, _cpos, _pos, _vel, _acc, _ep, _ev, _ea, np, ns);
			}
			if (xrm->get_num_generalized_coordinates())
			{
				unsigned int m_size = 0;
				unsigned int j_size = 0;
				fs.read((char*)&m_size, sizeof(unsigned int));
				fs.read((char*)&j_size, sizeof(unsigned int));
				/*if (pmrs.size() != m_size)
				{

				}
				if (kcrs.size() != j_size)
				{

				}*/
				xMultiBodyModel* xmbd = xdm->XMBDModel();
				
				for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
				{
					struct_pmr pr = { 0, };
					struct_pmr* _pmr = xrm->get_mass_result_ptr(it.key().toStdString());
					if (_pmr == NULL)
					{
						emit result_call_send_progress(-1, "!_Body that exists in the current model does not exist in the results.");
						xdm->release_result_manager();
						break;
					}						
					fs.read((char*)&pr, sizeof(struct_pmr));
					_pmr[cnt] = pr;
				}
				for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
				{
					struct_kcr kr = { 0, };
					struct_kcr* _kcr = xrm->get_joint_result_ptr(it.key().toStdString());
					if (_kcr == NULL)
					{
						emit result_call_send_progress(-1, "!_Joint that exists in the current model does not exist in the results.");
						xdm->release_result_manager();
						break;
					}
					fs.read((char*)&kr, sizeof(struct_kcr));
					_kcr[cnt] = kr;
				}
				for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
				{
					struct_kcr kr = { 0, };
					struct_kcr* _kcr = xrm->get_joint_result_ptr(it.key().toStdString());
					if (_kcr == NULL)
					{
						emit result_call_send_progress(-1, "!_Driving that exists in the current model does not exist in the results.");
						xdm->release_result_manager();
						break;
					}
					fs.read((char*)&kr, sizeof(struct_kcr));
					_kcr[cnt] = kr;
				}
			}
			cnt++;
			fs.close();
			emit result_call_send_progress(10 + cnt, "");
		}

		xvAnimationController::allocTimeMemory(xdm->XResult()->get_num_parts());
		double* _time = xrm->get_times();
		for (unsigned int i = 0; i < xrm->get_num_parts(); i++)
			xvAnimationController::addTime(i, static_cast<float>(_time[i]));
		xvAnimationController::setTotalFrame(is_terminated ? xrm->get_terminated_num_parts() - 1 : xrm->get_num_parts() - 1);
		if (_cpos) delete[] _cpos;
		if (_pos) delete[] _pos;
		if (_vel) delete[] _vel;
		if (_acc) delete[] _acc;
		if (_ep) delete[] _ep;
		if (_ev) delete[] _ev;
		if (_ea) delete[] _ea;
	}
	emit result_call_finish();
}