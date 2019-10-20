#include "xdynamics_manager/xResultManager.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "xColorControl.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <filesystem>

xResultManager::xResultManager()
	: time(NULL)
	, ptrs(NULL)
	, vtrs(NULL)
	, ctrs(NULL)
	, c_cluster_pos(NULL)
	, c_particle_pos(NULL)
	, c_particle_vel(NULL)
	, c_particle_acc(NULL)
	, c_particle_ep(NULL)
	, c_particle_ev(NULL)
	, c_particle_ea(NULL)
	, c_generalized_coord_q(NULL)
	, c_generalized_coord_qd(NULL)
	, c_generalized_coord_q_1(NULL)
	, c_generalized_coord_rhs(NULL)
	, p2p_contact_count(NULL)
	, p2p_contact_id(NULL)
	, p2p_contact_tsd(NULL)
	, p2pl_contact_count(NULL)
	, p2pl_contact_id(NULL)
	, p2pl_contact_tsd(NULL)
	, p2cyl_contact_count(NULL)
	, p2cyl_contact_id(NULL)
	, p2cyl_contact_tsd(NULL)
	, p2tri_contact_count(NULL)
	, p2tri_contact_id(NULL)
	, p2tri_contact_tsd(NULL)
	, nparticles(0)
	, nclusters(0)
	, ngeneralized_coordinates(0)
	, nconstraints(0)
	, terminated_num_parts(0)
	, nparts(0)
	, ncparts(0)
{

}

xResultManager::~xResultManager()
{
	if (time) delete[] time; time = NULL;
	if (ptrs) delete[] ptrs; ptrs = NULL;
	if (vtrs) delete[] vtrs; vtrs = NULL;
	if (ctrs) delete[] ctrs; ctrs = NULL;
	if(pmrs.size()) pmrs.delete_all();
	if(kcrs.size()) kcrs.delete_all();
}

void xResultManager::xRun(const std::string _cpath, const std::string _cname)
{
	char cmd[64] = { 0, };
	
	cur_path = _cpath;
	cur_name = _cname;
	std::string _path = _cpath + _cname + "/";
	std::cout << "Welcome to result world." << std::endl;
	std::cout << "Current path - " << _path.c_str() << std::endl;

	xUtilityFunctions::DirectoryFileList(_path.c_str());
	int ret = 0;
	int ncmd = 0;
	while (1)
	{
		ret = 0;
		std::cout << ">> ";
		std::cin.getline(cmd, sizeof(cmd), '\n');
		ncmd = xstring(cmd).n_split_string(" ");
		switch (ncmd)
		{
		case 0: ret = Execute0(cmd); break;
		case 1: ret = Execute1(cmd); break;
		case 2: ret = Execute2(cmd); break;
		}
	
		if (!ret)
			std::cout << "The command you entered does not exist." << std::endl;
		else if (ret == -1)
			break;
		fflush(stdin);
	}
}

void xResultManager::set_num_parts(unsigned int npt)
{
	nparts = npt;
}

unsigned int xResultManager::get_num_parts()
{
	return nparts;
}

unsigned int xResultManager::get_num_particles()
{
	return nparticles;
}

unsigned int xResultManager::get_num_clusters()
{
	return nclusters;
}

unsigned int xResultManager::get_num_generalized_coordinates()
{
	return ngeneralized_coordinates;
}

unsigned int xResultManager::get_num_constraint_equations()
{
	return nconstraints;
}

unsigned int xResultManager::get_terminated_num_parts()
{
	return terminated_num_parts;
}

unsigned int xResultManager::get_current_part_number()
{
	return ncparts;
}

double * xResultManager::get_times()
{
	return time;
}

void xResultManager::export_mass_result_to_text(std::string name)
{
	xmap<xstring, struct_pmr*>::iterator it = pmrs.find(name);
	std::fstream ifs;
	std::string new_file_name = xModel::makeFilePath(name + ".txt");
	ifs.open(new_file_name, ios::out);
	ifs << "time " << "px " << "py " << "pz " << "vx " << "vy " << "vz " 
		<< "ep0 " << "ep1 " << "ep2 " << "ep3 " << "ev0 " << "ev1 " << "ev2 " << "ev3 " 
		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
		<< "cfx " << "cfy " << "cfz " << "cmx " << "cmy " << "cmz "
		<< "hfx " << "hfy " << "hfz " << "hmx " << "hmy " << "hmz "
		<< "em0 " << "em1 " << "em2 " << "em3 "
		<< "avx " << "avy " << "avz " << "aax " << "aay " << "aaz "
		<< "ax " << "ay " << "az "
		<< "ea0 " << "ea1 " << "ea2 " << "ea3" << std::endl;
	struct_pmr* _pmr = it.value();
	for(unsigned int i = 0 ; i < ncparts; i++)//foreach(xPointMass::pointmass_result pr, *rst)
	{
		ifs << time[i] << " ";
		for (unsigned int j = 0; j < 49; j++)
		{
			double v = *(&(_pmr[i].pos.x) + j);
			ifs << v << " ";
		}
		ifs << std::endl;
	}
	ifs.close();
}

void xResultManager::export_joint_result_to_text(std::string n)
{
	xmap<xstring, struct_kcr*>::iterator it = kcrs.find(n);
	std::fstream ifs;
	std::string new_file_name = xModel::makeFilePath(n + ".txt");
	ifs.open(new_file_name, ios::out);
	ifs << "time " << "locx " << "locy " << "locz "
		<< "iafx " << "iafy " << "iafz " << "irfx " << "irfy " << "irfz "
		<< "jafx " << "jafy " << "jafz " << "jrfx " << "jrfy " << "jrfz ";
	struct_kcr* _kcr = it.value();
	for (unsigned int i = 0; i < ncparts; i++)
	{
		ifs << time[i] << " ";
		for (unsigned int j = 0; j < 15; j++)
		{
			double v = *(&(_kcr[i].location.x) + j);
			ifs << v << " ";
		}
		ifs << std::endl;
	}
	ifs.close();
}

void xResultManager::setup_particle_buffer_color_distribution(xColorControl* xcc, int sframe, int cframe)
{
	if (ctrs)
	{
		xColorControl::ColorMapType cmt = xcc->Target();
		if (!xcc->isUserLimitInput())
			xcc->setMinMax(get_min_result_value(cmt), get_max_result_value(cmt));
		xcc->setLimitArray();
		unsigned int neach = static_cast<unsigned int>(nparticles / nclusters);
		for (int i = 0; i <= ncparts; i++)
		{
			unsigned int idx = nparticles * i;
			unsigned int sidx = nclusters * i;
			float *pbuf = ptrs + idx * 4;
			float *vbuf = vtrs + sidx * 3;
			float *cbuf = ctrs + idx * 4;
			for (unsigned int j = 0; j < nparticles; j++)
			{
				unsigned int sj = (j / neach);
				xcc->getColorRamp(pbuf + j * 4, vbuf + sj * 3, cbuf + j * 4);
				cbuf[j * 4 + 3] = 1.0;
			}
		}
	}
}

void xResultManager::set_current_part_number(unsigned int cnpt)
{
	ncparts = cnpt;
}

bool xResultManager::initialize_from_exist_results(std::string path)
{
	std::string file_path;
	stringstream ss(file_path);
	ss << path << xModel::getModelName() << ".cdn";
	std::fstream fs;
	fs.open(ss.str(), std::ios::in | std::ios::binary);
	//unsigned int _npart = 0;
	
	//ss << path << "/part" << setw(4) << setfill('0') << _npart++ << ".bin";
	//std::fstream fs;
	//fs.open(ss.str(), std::ios::in | std::ios::binary);
	/*double ct = 0.0;
	char id = '0';*/
	//fs.read((char*)&_nstep, sizeof(unsigned int));
	fs.read((char*)&nparts, sizeof(unsigned int));
	alloc_time_momory(nparts);
	while (!fs.eof())
	{
		char id = '0';
		fs.read(&id, sizeof(char));
		switch (id)
		{
		case 'm':
			fs.read((char*)&ngeneralized_coordinates, sizeof(unsigned int));
			fs.read((char*)&nconstraints, sizeof(unsigned int));
			break;
		case 'd':
			fs.read((char*)&nparticles, sizeof(unsigned int));
			fs.read((char*)&nclusters, sizeof(unsigned int));
			alloc_dem_result_memory(nparticles, nclusters);
			break;
		}
	}
	
	//char id = '0';
	//fs.open(&id, sizeof(char));
	//
	//	_npart++;
	//	fs.read((char*)&ct, sizeof(double));
	//	fs.read(&id, sizeof(char));
	//	if (id == 'd')
	//	{
	//		unsigned int _np = 0;
	//		unsigned int _nc = 0;
	//		fs.read((char*)&_np, sizeof(unsigned int));
	//		fs.read((char*)&_nc, sizeof(unsigned int));
	//		
	//	}
	//	if (nparticles)
	//	{
	//		char id = 'd';
	//		qf.write(&id, sizeof(char));
	//		qf.write((char*)&nparticles, sizeof(unsigned int));
	//		qf.write((char*)&nclusters, sizeof(unsigned int));
	//		qf.write((char*)c_particle_pos, sizeof(double) * nparticles * 4);
	//		qf.write((char*)c_particle_vel, sizeof(double) * nclusters * 3);
	//		qf.write((char*)c_particle_acc, sizeof(double) * nclusters * 3);
	//		qf.write((char*)c_particle_ep, sizeof(double) * nclusters * 4);
	//		qf.write((char*)c_particle_ev, sizeof(double) * nclusters * 4);
	//		qf.write((char*)c_particle_ea, sizeof(double) * nclusters * 4);
	//		//qf.write((char*)avel, sizeof(double) * ns * 4);
	//		if ((nparticles != nclusters) && c_cluster_pos)
	//			qf.write((char*)c_cluster_pos, sizeof(double) * nclusters * 4);
	//	}
	//	if (ngeneralized_coordinates)
	//	{
	//		char id = 'm';
	//		unsigned int mdim = ngeneralized_coordinates + xModel::OneDOF();
	//		unsigned int tdim = ngeneralized_coordinates + nconstraints;
	//		qf.write(&id, sizeof(char));
	//		qf.write((char*)c_generalized_coord_q, sizeof(double) * mdim);
	//		qf.write((char*)c_generalized_coord_qd, sizeof(double) * mdim);
	//		qf.write((char*)c_generalized_coord_q_1, sizeof(double) * mdim);
	//		qf.write((char*)c_generalized_coord_rhs, sizeof(double) * tdim);
	//	}
	//}
	return true;
}

bool xResultManager::upload_exist_results(std::string path)
{
	std::string file_path;
	stringstream ss(file_path);
	unsigned int _nparts = 0;
//	ss << path << "/part" << setw(4) << setfill('0') << _npart++ << ".bin";
	//list<std::string> file_list;
	while (1)
	{
		ss.str("");
		ss << path << "part" << setw(4) << setfill('0') << _nparts << ".bin";
		
		filesystem::path p(ss.str());
		if (filesystem::exists(p))
		{
			flist.push_back(ss.str());
			_nparts++;
		}			
		else
			break;
	}
	if (nparts != _nparts)
	{

	}
	std::fstream fs;
	
	unsigned int cnt = 0;
	double ct = 0;
	double* _cpos = NULL, *_pos = NULL, *_vel = NULL, *_acc = NULL, *_ep = NULL, *_ev = NULL, *_ea = NULL;
	if (nparticles)
	{
		 _pos = new double[nparticles * 4];
		 _vel = new double[nclusters * 3];
		 _acc = new double[nclusters * 3];
		 _ep = new double[nclusters * 4];
		 _ev = new double[nclusters * 4];
		 _ea = new double[nclusters * 4];
		 if (nclusters != nparticles)
			 _cpos = new double[nclusters * 4];
	}
	for (xlist<xstring>::iterator it = flist.begin(); it != flist.end(); it.next())
	{
		fs.open(it.value().toStdString(), std::ios::in | std::ios::binary);
		fs.read((char*)&ct, sizeof(double));
		time[cnt] = ct;
		if (nparticles)
		{
			unsigned int _npt = 0;
			unsigned int _nct = 0;
			fs.read((char*)&_npt, sizeof(unsigned int));
			fs.read((char*)&_nct, sizeof(unsigned int));
			fs.read((char*)_pos, sizeof(double) * nparticles * 4);
			fs.read((char*)_vel, sizeof(double) * nclusters * 3);			
			fs.read((char*)_acc, sizeof(double) * nclusters * 3);
			fs.read((char*)_ep, sizeof(double) * nclusters * 4);
			fs.read((char*)_ev, sizeof(double) * nclusters * 4);
			fs.read((char*)_ea, sizeof(double) * nclusters * 4);
			if(nclusters != nparticles)
				fs.read((char*)_cpos, sizeof(double) * nclusters * 4);
			this->save_dem_result(cnt, _cpos, _pos, _vel, _acc, _ep, _ev, _ea, nparticles, nclusters);
		}
		if (ngeneralized_coordinates)
		{
			unsigned int m_size = 0;
			unsigned int j_size = 0;
			fs.read((char*)&m_size, sizeof(unsigned int));
			fs.read((char*)&j_size, sizeof(unsigned int));
			if (pmrs.size() != m_size)
			{

			}
			if (kcrs.size() != j_size)
			{

			}
			for (xmap<xstring, struct_pmr*>::iterator it = pmrs.begin(); it != pmrs.end(); it.next())
			{
				struct_pmr pr = { 0, };
				struct_pmr* _pmr = it.value();
				fs.read((char*)&pr, sizeof(struct_pmr));
				_pmr[cnt] = pr;
			}
			for (xmap<xstring, struct_kcr*>::iterator it = kcrs.begin(); it != kcrs.end(); it.next())
			{
				struct_kcr kr = { 0, };
				struct_kcr* _kcr = it.value();
				fs.read((char*)&kr, sizeof(struct_kcr));
				_kcr[cnt] = kr;
			}
			cnt++;
		}
		fs.close();
	}
	if (_cpos) delete[] _cpos;
	if (_pos) delete[] _pos;
	if (_vel) delete[] _vel;
	if (_acc) delete[] _acc;
	if (_ep) delete[] _ep;
	if (_ev) delete[] _ev;
	if (_ea) delete[] _ea;
	return true;
}

float xResultManager::get_min_result_value(xColorControl::ColorMapType cmt)
{
	float v = 0.0;
	switch (cmt)
	{
	case xColorControl::COLORMAP_POSITION_X: v = min_particle_position[0]; break;
	case xColorControl::COLORMAP_POSITION_Y: v = min_particle_position[1]; break;
	case xColorControl::COLORMAP_POSITION_Z: v = min_particle_position[2]; break;
	case xColorControl::COLORMAP_VELOCITY_X: v = min_particle_velocity[0]; break;
	case xColorControl::COLORMAP_VELOCITY_Y: v = min_particle_velocity[1]; break;
	case xColorControl::COLORMAP_VELOCITY_Z: v = min_particle_velocity[2]; break;
	case xColorControl::COLORMAP_POSITION_MAG: v = min_particle_position_mag; break;
	case xColorControl::COLORMAP_VELOCITY_MAG: v = min_particle_velocity_mag; break;
	}
	return v;
}

float xResultManager::get_max_result_value(xColorControl::ColorMapType cmt)
{
	float v = 0.0;
	switch (cmt)
	{
	case xColorControl::COLORMAP_POSITION_X: v = max_particle_position[0]; break;
	case xColorControl::COLORMAP_POSITION_Y: v = max_particle_position[1]; break;
	case xColorControl::COLORMAP_POSITION_Z: v = max_particle_position[2]; break;
	case xColorControl::COLORMAP_VELOCITY_X: v = max_particle_velocity[0]; break;
	case xColorControl::COLORMAP_VELOCITY_Y: v = max_particle_velocity[1]; break;
	case xColorControl::COLORMAP_VELOCITY_Z: v = max_particle_velocity[2]; break;
	case xColorControl::COLORMAP_POSITION_MAG: v = max_particle_position_mag; break;
	case xColorControl::COLORMAP_VELOCITY_MAG: v = max_particle_velocity_mag; break;
	}
	return v;
}

xlist<xstring>* xResultManager::get_part_file_list()
{
	return &flist;
}

struct_pmr * xResultManager::get_mass_result_ptr(std::string n)
{
	xmap<xstring, struct_pmr*>::iterator it = pmrs.find(n);
	if (it != pmrs.end())
		return it.value();
	return NULL;
}

struct_kcr * xResultManager::get_joint_result_ptr(std::string n)
{
	xmap<xstring, struct_kcr*>::iterator it = kcrs.find(n);
	if (it != kcrs.end())
		return it.value();
	return NULL;
}

xmap<xstring, struct_pmr*>* xResultManager::get_mass_result_xmap()
{
	return &pmrs;
}

xmap<xstring, struct_kcr*>* xResultManager::get_joint_result_xmap()
{
	return &kcrs;
}

xmap<xstring, xDrivingRotationResultData>* xResultManager::get_rotation_driving_result_xmap()
{
	return &xdrr;
}

xlist<unsigned int>* xResultManager::get_distribution_id_list()
{
	return &dist_id;
}

float * xResultManager::get_particle_position_result_ptr()
{
	return ptrs;
}

float * xResultManager::get_particle_velocity_result_ptr()
{
	return vtrs;
}

float * xResultManager::get_cluster_position_result_ptr()
{
	return NULL;// c_cluster_pos;
}

float * xResultManager::get_particle_color_result_ptr()
{
	return ctrs;
}

void xResultManager::set_gpu_process_device(bool b)
{
	is_gpu_process = b;
}

void xResultManager::set_num_generailzed_coordinates(unsigned int ng)
{
	ngeneralized_coordinates = ng;
}

void xResultManager::set_num_constraints_equations(unsigned int nc)
{
	nconstraints = nc;
}

void xResultManager::set_distribution_result(std::list<unsigned int> dl)
{
	if (dist_id.size())
	{
		dist_id.remove_all();
	}
	for (std::list<unsigned int>::iterator it = dl.begin(); it != dl.end(); it++)
	{
		dist_id.push_back(*it);
	}
}

void xResultManager::set_terminated_num_parts(unsigned int _npt)
{
	terminated_num_parts = _npt;
}

void xResultManager::set_p2p_contact_data(int n)
{
	p2p_contact = n;
}

void xResultManager::set_p2pl_contact_data(int n)
{
	p2pl_contact = n;
}

void xResultManager::set_p2cyl_contact_data(int n)
{
	p2cyl_contact = n;
}

void xResultManager::set_p2tri_contact_data(int n)
{
	p2tri_contact = n;
}

bool xResultManager::alloc_time_momory(unsigned int npart)
{
	if (time) delete[] time; time = NULL;
	time = new double[npart];
	return true;
}

bool xResultManager::alloc_dem_result_memory(unsigned int np, unsigned int ns)
{
	nparticles = np;
	nclusters = ns;
	if (!nparticles)
		return false;
	if (ptrs) delete[] ptrs; ptrs = NULL;
	if (vtrs) delete[] vtrs; vtrs = NULL;
	if (ctrs) delete[] ctrs; ctrs = NULL;

	ptrs = new float[nparts * nparticles * 4]; 
	for (unsigned int i = 0; i < nparts * nparticles * 4; i++) 
		ptrs[i] = 0.f;
		//memset(ptrs, 0, sizeof(float) * nparts * nparticles * 4);
	vtrs = new float[nparts * nclusters * 3]; 
	for (unsigned int i = 0; i < nparts * nclusters * 3; i++) 
		vtrs[i] = 0.f;
		//memset(vtrs, 0, sizeof(float) * nparts * nclusters * 3);
	ctrs = new float[nparts * nparticles * 4]; 
	for (unsigned int i = 0; i < nparts * nparticles * 4; i++)
		ctrs[i] = 0.f;
		//memset(ctrs, 0, sizeof(float) * nparts * nclusters * 4);
	//throw runtime_error("error");
	allocated_size += nparts * (nparticles * 4 + nclusters * 3 + nparticles * 4) * sizeof(float);
	//memset(ptrs, 0, allocated_size);
	return true;
}

bool xResultManager::alloc_mass_result_memory(std::string name)
{
	if (!nparts)
		return false;
	xmap<xstring, struct_pmr*>::iterator it = pmrs.find(name);
	if (it != pmrs.end())
		return false;
	struct_pmr* spmrs = new struct_pmr[nparts];
	pmrs.insert(name, spmrs);
	return true;
}

bool xResultManager::alloc_joint_result_memory(std::string name)
{
	if (!nparts)
		return false;
	xmap<xstring, struct_kcr*>::iterator it = kcrs.find(name);
	if (it != kcrs.end())
		return false;
	struct_kcr* skcrs = new struct_kcr[nparts];
	kcrs.insert(name, skcrs);
	//xdrr.insert(name, { 0,0 });
	return true;
}

bool xResultManager::alloc_driving_rotation_result_memory(std::string name)
{
	xmap<xstring, xDrivingRotationResultData>::iterator it = xdrr.find(name);
	if (it != xdrr.end())
		return false;
	xdrr.insert(name, { 0, });
	return true;
}

bool xResultManager::save_dem_result(
	unsigned int i, double* cpos, double * pos, double * vel, double* acc, 
	double* ep, double* ev, double* ea, unsigned int np, unsigned int ns)
{
	c_particle_pos = pos;
	c_particle_vel = vel;
	c_particle_acc = acc;
	c_particle_ep = ep;
	c_particle_ev = ev;
	c_particle_ea = ea;
	c_cluster_pos = cpos;
	unsigned int neach = 1;
	unsigned int sid = 0;
	unsigned int vid = 0;
	if (nparticles != np)
	{
		return false;
	}
	if (np != ns)
		neach = np / ns;
	sid = np * i * 4;
	vid = ns * i * 3;
	for (unsigned int j = 0; j < np; j++)
	{
		unsigned int s = j * 4;

		vector4f p = new_vector4f(
			static_cast<float>(pos[s + 0]),
			static_cast<float>(pos[s + 1]),
			static_cast<float>(pos[s + 2]),
			static_cast<float>(pos[s + 3]));

		s += sid;
		ptrs[s + 0] = p.x;// static_cast<float>(_pos[s + 0]);
		ptrs[s + 1] = p.y;// static_cast<float>(_pos[s + 1]);
		ptrs[s + 2] = p.z;// static_cast<float>(_pos[s + 2]);
		ptrs[s + 3] = p.w;// static_cast<float>(_pos[s + 3]);
		if (max_particle_position[0] < p.x) max_particle_position[0] = p.x;// buffers[s + 0];
		if (max_particle_position[1] < p.y) max_particle_position[1] = p.y;// buffers[s + 1];
		if (max_particle_position[2] < p.z) max_particle_position[2] = p.z;// buffers[s + 2];

		if (min_particle_position[0] > p.x) min_particle_position[0] = p.x;// buffers[s + sid + 0];
		if (min_particle_position[1] > p.y) min_particle_position[1] = p.y;// buffers[s + sid + 1];
		if (min_particle_position[2] > p.z) min_particle_position[2] = p.z;// buffers[s + sid + 2];
		float p_mag = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
		if (max_particle_position_mag < p_mag) max_particle_position_mag = p_mag;
		if (min_particle_position_mag > p_mag) min_particle_position_mag = p_mag;
		vector3f vv = new_vector3f(0, 0, 0);
		unsigned int v = j * 3;
		if (np != ns)
			v = (j / neach) * 3;
		vv = new_vector3f(
			static_cast<float>(vel[v + 0]),
			static_cast<float>(vel[v + 1]),
			static_cast<float>(vel[v + 2]));

		v += vid;

		vtrs[v + 0] = vv.x;// static_cast<float>(_vel[v + 0]);
		vtrs[v + 1] = vv.y;// static_cast<float>(_vel[v + 1]);
		vtrs[v + 2] = vv.z;// static_cast<float>(_vel[v + 2]);

		if (max_particle_velocity[0] < vv.x) max_particle_velocity[0] = vv.x;// vbuffers[v + vid + 0];
		if (max_particle_velocity[1] < vv.y) max_particle_velocity[1] = vv.y;// vbuffers[v + vid + 1];
		if (max_particle_velocity[2] < vv.z) max_particle_velocity[2] = vv.z;// vbuffers[v + vid + 2];

		if (min_particle_velocity[0] > vv.x) min_particle_velocity[0] = vv.x;// vbuffers[v + vid + 0];
		if (min_particle_velocity[1] > vv.y) min_particle_velocity[1] = vv.y;// vbuffers[v + vid + 1];
		if (min_particle_velocity[2] > vv.z) min_particle_velocity[2] = vv.z;// vbuffers[v + vid + 2];

		float v_mag = sqrt(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z);// [v + 0] * vbuffers[v + 0] + vbuffers[v + 1] * vbuffers[v + 1] + vbuffers[v + 2] * vbuffers[v + 2]);

		if (max_particle_velocity_mag < v_mag) max_particle_velocity_mag = v_mag;
		if (min_particle_velocity_mag > v_mag) min_particle_velocity_mag = v_mag;
		ctrs[s + 0] = 0.0f;
		ctrs[s + 1] = 0.0f;
		ctrs[s + 2] = 1.0f;
		ctrs[s + 3] = 1.0f;
	}
	return true;
}

bool xResultManager::save_mass_result(unsigned int i, xPointMass * pm)
{
	struct_pmr pr = { 0, };
	//double *ptr = &pm->Positioon
	memcpy(&pr, pm->getPositionPointer(), sizeof(double) * 49);
	xmap<xstring, struct_pmr*>::iterator it = pmrs.find(pm->Name());
	if (it != pmrs.end())
	{
		struct_pmr* _pmr = it.value();
		_pmr[i] = pr;
		c_struct_pmr = pr;
	}
	else
		return false;
	return true;
}

bool xResultManager::save_joint_result(unsigned int i, std::string nm, struct_kcr _kcr)
{
	xmap<xstring, struct_kcr*>::iterator it = kcrs.find(nm);
	if (it != kcrs.end())
	{
		struct_kcr* __kcr = it.value();
		__kcr[i] = _kcr;
		c_struct_kcr = _kcr;
	}
	else
		return false;
	return true;
}

bool xResultManager::save_generalized_coordinate_result(double * q, double * qd, double * q_1, double * rhs)
{
	c_generalized_coord_q = q;
	c_generalized_coord_qd = qd;
	c_generalized_coord_q_1 = q_1;
	c_generalized_coord_rhs = rhs;
	return true;
}

bool xResultManager::save_p2p_contact_data(unsigned int * count, unsigned int * id, double * tsd)
{
	p2p_contact_count = count;
	p2p_contact_id = id;
	p2p_contact_tsd = tsd;
	return true;
}

bool xResultManager::save_p2pl_contact_data(unsigned int * count, unsigned int * id, double * tsd)
{
	p2pl_contact_count = count;
	p2pl_contact_id = id;
	p2pl_contact_tsd = tsd;
	return true;
}

bool xResultManager::save_p2cyl_contact_data(unsigned int * count, unsigned int * id, double * tsd)
{
	p2cyl_contact_count = count;
	p2cyl_contact_id = id;
	p2cyl_contact_tsd = tsd;
	return true;
}

bool xResultManager::save_p2tri_contact_data(unsigned int * count, unsigned int * id, double * tsd)
{
	p2tri_contact_count = count;
	p2tri_contact_id = id;
	p2tri_contact_tsd = tsd;
	return true;
}

void xResultManager::save_driving_rotation_result(unsigned int i, std::string nm, unsigned int n_rev, unsigned int dn_rev, double theta)
{
	xmap<xstring, xDrivingRotationResultData>::iterator it = xdrr.find(nm);
	it.setValue({ n_rev, dn_rev, theta });
}

bool xResultManager::export_step_data_to_file(unsigned int pt, double ct)
{
	time[pt] = ct;
	std::string file_name;
	stringstream ss(file_name);
	ss << (xModel::path + xModel::name).toStdString() << "/part" << setw(4) << setfill('0') << pt << ".bin";
	flist.push_back(ss.str());
	std::fstream qf;
	qf.open(ss.str(), std::ios::binary | std::ios::out);
	if (qf.is_open())
	{
		qf.write((char*)&ct, sizeof(double));
		if (nparticles)
		{
			qf.write((char*)&nparticles, sizeof(unsigned int));
			qf.write((char*)&nclusters, sizeof(unsigned int));
			qf.write((char*)c_particle_pos, sizeof(double) * nparticles * 4);
			qf.write((char*)c_particle_vel, sizeof(double) * nclusters * 3);
			qf.write((char*)c_particle_acc, sizeof(double) * nclusters * 3);
			qf.write((char*)c_particle_ep, sizeof(double) * nclusters * 4);
			qf.write((char*)c_particle_ev, sizeof(double) * nclusters * 4);
			qf.write((char*)c_particle_ea, sizeof(double) * nclusters * 4);
			if ((nparticles != nclusters) && c_cluster_pos)
				qf.write((char*)c_cluster_pos, sizeof(double) * nclusters * 4);
		}
		if (ngeneralized_coordinates)
		{
			unsigned int m_size = pmrs.size();
			unsigned int j_size = kcrs.size();
			unsigned int d_size = xdrr.size();
			qf.write((char*)&m_size, sizeof(unsigned int));
			qf.write((char*)&j_size, sizeof(unsigned int));
			qf.write((char*)&d_size, sizeof(unsigned int));
			for (xmap<xstring, struct_pmr*>::iterator it = pmrs.begin(); it != pmrs.end(); it.next())
			{
				struct_pmr* _pmr = it.value();
				qf.write((char*)&_pmr[pt], sizeof(struct_pmr));
			}
			for (xmap<xstring, struct_kcr*>::iterator it = kcrs.begin(); it != kcrs.end(); it.next())
			{
				struct_kcr* _kcr = it.value();
				qf.write((char*)&_kcr[pt], sizeof(struct_kcr));
			}
			for (xmap<xstring, xDrivingRotationResultData>::iterator it = xdrr.begin(); it != xdrr.end(); it.next())
			{
				qf.write((char*)&it.value(), sizeof(xDrivingRotationResultData));
			}
			unsigned int mdim = ngeneralized_coordinates + xModel::OneDOF();
			unsigned int tdim = ngeneralized_coordinates + nconstraints;
			//qf.write(&id, sizeof(char));
			qf.write((char*)c_generalized_coord_q, sizeof(double) * mdim);
			qf.write((char*)c_generalized_coord_qd, sizeof(double) * mdim);
			qf.write((char*)c_generalized_coord_q_1, sizeof(double) * mdim);
			qf.write((char*)c_generalized_coord_rhs, sizeof(double) * tdim);
		}	
		if (p2p_contact)
		{
			qf.write((char*)p2p_contact_count, sizeof(unsigned int) * nparticles);
			qf.write((char*)p2p_contact_id, sizeof(unsigned int) * nparticles * p2p_contact);
			qf.write((char*)p2p_contact_tsd, sizeof(double) * 2 * nparticles * p2p_contact);
		}
		if (p2pl_contact)
		{
			qf.write((char*)p2pl_contact_count, sizeof(unsigned int) * nparticles);
			qf.write((char*)p2pl_contact_id, sizeof(unsigned int) * nparticles * p2pl_contact);
			qf.write((char*)p2pl_contact_tsd, sizeof(double) * 2 * nparticles * p2pl_contact);
		}
		if (p2cyl_contact)
		{
			qf.write((char*)p2cyl_contact_count, sizeof(unsigned int) * nparticles);
			qf.write((char*)p2cyl_contact_id, sizeof(unsigned int) * nparticles * p2cyl_contact);
			qf.write((char*)p2cyl_contact_tsd, sizeof(double) * 2 * nparticles * p2cyl_contact);
		}
		if (p2tri_contact)
		{
			qf.write((char*)p2tri_contact_count, sizeof(unsigned int) * nparticles);
			qf.write((char*)p2tri_contact_id, sizeof(unsigned int) * nparticles * p2tri_contact);
			qf.write((char*)p2tri_contact_tsd, sizeof(double) * 2 * nparticles * p2tri_contact);
		}
	}
	qf.close();
	c_cluster_pos = NULL;
	c_particle_pos = NULL;
	c_particle_vel = NULL;
	c_particle_acc = NULL;
	c_particle_ep = NULL;
	c_particle_ev = NULL;
	c_particle_ea = NULL;

	c_generalized_coord_q = NULL;
	c_generalized_coord_qd = NULL;
	c_generalized_coord_q_1 = NULL;
	c_generalized_coord_rhs = NULL;
	return true;
}

void xResultManager::setCurrentPath(std::string new_path)
{
	cur_path = new_path;// wsprintfW(cur_path, TEXT("%s"), new_path);
}

void xResultManager::setCurrentName(std::string new_name)
{
	cur_name = new_name;// wsprintfW(cur_name, TEXT("%s"), new_name);
}

void xResultManager::ExportBPM2TXT(std::string& file_name)
{
	//std::fstream ifs;
	//ifs.open(file_name.c_str(), ios::binary | ios::in);
	////xLog::log(xUtilityFunctions::WideChar2String(file_name.c_str()));
	//int identifier;
	//unsigned int nr_part;
	//char t;
	//ifs.read((char*)&identifier, sizeof(int));
	//ifs.read((char*)&t, sizeof(char));
	//ifs.read((char*)&nr_part, sizeof(unsigned int));
	//xPointMass::pointmass_result *pms = new xPointMass::pointmass_result[nr_part];
	//ifs.read((char*)pms, sizeof(xPointMass::pointmass_result) * nr_part);
	//ifs.close();
	////size_t begin = file_name.find_last_of(".");
	//size_t end = file_name.find_last_of("/");
	//std::string new_file_name = file_name.substr(0, end + 1) + xUtilityFunctions::GetFileName(file_name.c_str()) + ".txt";
	//ifs.open(new_file_name, ios::out);
	//ifs << "time " << "px " << "py " << "pz " << "vx " << "vy " << "vz " << "ax " << "ay " << "az "
	//	<< "avx " << "avy " << "avz " << "aax " << "aay " << "aaz "
	//	<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
	//	<< "cfx " << "cfy " << "cfz " << "cmx " << "cmy " << "cmz "
	//	<< "hfx " << "hfy " << "hfz " << "hmx " << "hmy " << "hmz "
	//	<< "ep0 " << "ep1 " << "ep2 " << "ep3 "
	//	<< "ev0 " << "ev1 " << "ev2 " << "ev3 "
	//	<< "ea0 " << "ea1 " << "ea2 " << "ea3" << std::endl;

	//for (unsigned int i = 0; i < nr_part; i++)
	//{
	//	for (unsigned int j = 0; j < 46; j++)
	//	{
	//		double v = *(&(pms[0].time) + i * 46 + j);
	//		ifs << v << " ";
	//	}
	//	ifs << std::endl;
	//}
	//ifs.close();
}

void xResultManager::ExportBKC2TXT(std::string& file_name)
{
	//std::fstream ifs;
	//ifs.open(file_name, ios::binary | ios::in);
	//int identifier;
	//unsigned int nr_part;
	//char t;
	//ifs.read((char*)&identifier, sizeof(int));
	//ifs.read((char*)&t, sizeof(char));
	//ifs.read((char*)&nr_part, sizeof(unsigned int));
	//xKinematicConstraint::kinematicConstraint_result *pms = new xKinematicConstraint::kinematicConstraint_result[nr_part];
	//ifs.read((char*)pms, sizeof(xKinematicConstraint::kinematicConstraint_result) * nr_part);
	//ifs.close();
	////size_t begin = file_name.find_last_of(".");
	//std::string new_file_name = xUtilityFunctions::GetFileName(file_name.c_str()) + ".txt";//.substr(0, begin) + ".txt";
	//ifs.open(new_file_name, ios::out);
	//ifs << "time " << "locx " << "locy " << "locz "// << "vx " << "vy " << "vz " << "ax " << "ay " << "az "
	//	<< "iafx " << "iafy " << "iafz " << "irfx " << "irfy " << "irfz "
	//	<< "jafx " << "jafy " << "jafz " << "jrfx " << "jrfy " << "jrfz ";

	//for (unsigned int i = 0; i < nr_part; i++)
	//{
	//	for (unsigned int j = 0; j < 16; j++)
	//	{
	//		double v = *(&(pms[0].time) + i * 16 + j);
	//		ifs << v << " ";
	//	}
	//	ifs << std::endl;
	//}
	//ifs.close();
}

int xResultManager::Execute0(char *d)
{
	return 1;
}

int xResultManager::Execute1(char *d)
{
	if (!strcmp("exit", d))
		return -1;
	else if (!strcmp("list", d))
	{
		xUtilityFunctions::DirectoryFileList((cur_path + cur_name).toStdString().c_str());
		return 1;
	}
	return 0;
}

int xResultManager::Execute2(char *d)
{
	char val[64] = { 0, };
	std::string data[2];
	xstring x = d;
	x.split(" ", 2, data);
	if (data[0] == "get")
	{
		if (data[1] == "ascii")
		{
			std::cout << "Please enter a result file to import : ";
			std::cin >> val;
			std::string fn = (cur_path + cur_name + "/").toStdString() + val;
			if (xUtilityFunctions::ExistFile(fn.c_str()))
			{
				std::string ext = xUtilityFunctions::FileExtension(fn.c_str());
				if (ext == ".bpm")
					ExportBPM2TXT(fn);
				else if (ext == ".bkc")
					ExportBKC2TXT(fn);
			}
			return 1;
		}
	}
	else if (data[0] == "set")
	{
		if (data[1] == "mode")
		{
			std::cout << "Please enter a model name : ";
			std::cin >> val;
			std::string cname = val;
			std::string fn = cur_path.toStdString() + val + "/";
			if (xUtilityFunctions::ExistFile(fn.c_str()))
			{
				cur_name = cname;
			//	_path = fn;
				xUtilityFunctions::DirectoryFileList(fn.c_str());
			}
			else
			{
				std::cout << "The model you entered does not exist." << std::endl;
			}
			return 1;
		}
	}
	return 0;
}

