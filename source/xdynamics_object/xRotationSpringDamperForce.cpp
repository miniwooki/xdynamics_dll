#include "xdynamics_object/xRotationSpringDamperForce.h"
#include "xdynamics_manager/xDynamicsManager.h"
#include "xdynamics_simulation/xSimulation.h"

xRotationSpringDamperForce::xRotationSpringDamperForce()
	: xForce()
	, init_theta(0.0)
	, k(0.0)
	, c(0.0)
	, L(new_vector3d(0.0, 0.0, 0.0))
	, f(0.0)
	, l(0.0)
	, dl(0.0)
	, nrdci(0)
	, nConnection(0)
	, xrdci(NULL)
	, connection_data(NULL)
	, kc_value(NULL)
	, free_angle(NULL)
	, attached_body_info(NULL)
	, attached_body_data(NULL)
	, udrl(0)
	, n_rev(0)
	, theta(0)
{

}

xRotationSpringDamperForce::xRotationSpringDamperForce(std::string _name)
	: xForce(_name, RSDA)
	, init_theta(0.0)
	, k(0.0)
	, c(0.0)
	, L(new_vector3d(0.0, 0.0, 0.0))
	, f(0.0)
	, l(0.0)
	, dl(0.0)
	, nrdci(0)
	, nConnection(0)
	, xrdci(NULL)
	, connection_data(NULL)
	, kc_value(NULL)
	, free_angle(NULL)
	, attached_body_info(NULL)
	, attached_body_data(NULL)
	, udrl(0)
	, n_rev(0)
	, theta(0)
{

}

xRotationSpringDamperForce::~xRotationSpringDamperForce()
{
	if (xrdci) delete[] xrdci; xrdci = NULL;
	if (connection_data) delete[] connection_data; connection_data = NULL;
	if (attached_body_info) delete[] attached_body_info; attached_body_info = NULL;
	if (attached_body_data) delete[] attached_body_data; attached_body_data = NULL;
	if (kc_value) delete[] kc_value; kc_value = NULL;
	if (free_angle) delete[] free_angle; free_angle = NULL;
}

void xRotationSpringDamperForce::SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xRSDAData& d)
{
	xForce::i_ptr = ip;
	xForce::j_ptr = jp;
	loc = new_vector3d(d.lx, d.ly, d.lz);
	f_i = new_vector3d(d.fix, d.fiy, d.fiz);
	g_i = new_vector3d(d.gix, d.giy, d.giz);
	f_j = new_vector3d(d.fjx, d.fjy, d.fjz);
	g_j = new_vector3d(d.gjx, d.gjy, d.gjz);
	init_theta = acos(dot(f_i, f_j));
	h_i = cross(f_i, g_i);// new_vector3d(d.uix, d.uiy, d.uiz);
	h_j = cross(f_j, g_j);// new_vector3d(d.ujx, d.ujy, d.ujz);
	k = d.k;
	c = d.c;
	init_theta = d.init_r;
	xForce::spi = i_ptr->toLocal(loc - i_ptr->Position());
	xForce::spj = j_ptr->toLocal(loc - j_ptr->Position());
}

void xRotationSpringDamperForce::SetupDataFromBodyAndPoint2Spring(xPointMass * ip, xPoint2Spring * ps, xRSDAData & d)
{
	xForce::i_ptr = ip;
	xForce::j_ptr = nullptr;
	xForce::g_ptr = ps;
	loc = new_vector3d(d.lx, d.ly, d.lz);
	f_i = new_vector3d(d.fix, d.fiy, d.fiz);
	g_i = new_vector3d(d.gix, d.giy, d.giz);
	f_j = new_vector3d(d.fjx, d.fjy, d.fjz);
	g_j = new_vector3d(d.gjx, d.gjy, d.gjz);
	init_theta = acos(dot(f_i, f_j));
	h_i = cross(f_i, g_i);// new_vector3d(d.uix, d.uiy, d.uiz);
	h_j = cross(f_j, g_j);// new_vector3d(d.ujx, d.ujy, d.ujz);
	k = d.k;
	c = d.c;
	init_theta = d.init_r;
	vector3d p2s_first_position = new_vector3d(ps->p0->x, ps->p0->y, ps->p0->z);
	xForce::spi = i_ptr->toLocal(loc - i_ptr->Position());
	xForce::spj = j_ptr->toLocal(loc - p2s_first_position);
}

void xRotationSpringDamperForce::SetupDataFromListData(xRSDAData&d, std::string data)
{
	xForce::i_ptr = NULL;
	xForce::j_ptr = NULL;
	std::fstream fs;
	fs.open(data, std::ios::in);
	unsigned int cnt = 0;
	std::string ch;
	std::list<xRSDAConnectionInformation> xrdcis;
	std::list<xRSDAConnectionData> clist;
	std::list<xSpringDamperCoefficient> cvalue;
	if (fs.is_open())
	{
		unsigned int id = 0;
		while (!fs.eof())
		{
			fs >> ch;
			if (ch == "kc_value")
			{
				unsigned int nkc = 0;
				fs >> nkc;
				for (unsigned int i = 0; i < nkc; i++)
				{
					xSpringDamperCoefficient xsc = { 0, };
					fs >> id >> xsc.k >> xsc.c;
					cvalue.push_back(xsc);
				}

			}
			else if (ch == "connection_list")
			{
				unsigned int nlist = 0;
				fs >> nlist;
				for (unsigned int i = 0; i < nlist; i++)
				{
					xRSDAConnectionInformation xsi = { 0, cnt, 0 };
					xRSDAConnectionData xsd = { 0, };

					fs >> xsi.id >> xsi.nrsda;
					xrdcis.push_back(xsi);
					unsigned int cid = 0;
					unsigned int kc = 0;
					for (unsigned int i = 0; i < xsi.nrsda; i++)
					{
						fs >> cid >> kc;
						xsd.jd = cid;
						xsd.kc_id = kc;
						clist.push_back(xsd);
					}
					cnt += xsi.nrsda;
				}
			}
			else if (ch == "mass_particle_connection_list")
			{
				std::string ch;
				size_t cnt = 0;
				fs >> nBodyAttached;
				attached_body_info = new xSpringDamperBodyConnectionInfo[nBodyAttached];
				list<xRSDABodyAttachedData> bc_list;
				for (unsigned int i = 0; i < nBodyAttached; i++)
				{
					fs >> ch >> ch;
					attached_body_info[i].cbody = ch.c_str();
					attached_body_info[i].sid = cnt;
					fs >> ch >> attached_body_info[i].nconnection;
					for (unsigned int j = 0; j < attached_body_info[i].nconnection; j++)
					{
						xRSDABodyAttachedData d = { 0, };
						fs >> d.id >> d.jd >> d.kc_id >> d.rx >> d.ry >> d.rz
							>> d.fix >> d.fiy >> d.fiz >> d.gix >> d.giy >> d.giz
							>> d.fjx >> d.fjy >> d.fjz >> d.gjx >> d.gjy >> d.gjz;
						bc_list.push_back(d);
					}
					cnt = bc_list.size();
				}
				cnt = 0;
				nBodyAttachedData = bc_list.size();
				if (nBodyAttachedData)
				{
					attached_body_data = new xRSDABodyAttachedData[nBodyAttachedData];
					for (list<xRSDABodyAttachedData>::iterator d = bc_list.begin(); d != bc_list.end(); d++)
					{
						attached_body_data[cnt++] = *d;
					}
				}
			}
		}
	}
	fs.close();
	if (!kc_value)
	{
		unsigned int ct = 0;
		kc_value = new xSpringDamperCoefficient[cvalue.size()];
		for (list<xSpringDamperCoefficient>::iterator c = cvalue.begin(); c != cvalue.end(); c++)
			//foreach(xSpringDamperCoefficient c, cvalue)
		{
			kc_value[ct++] = *c;
		}
	}
	if (!xrdci)
	{
		xrdci = new xRSDAConnectionInformation[xrdcis.size()];
		unsigned int ct = 0;
		for (list<xRSDAConnectionInformation>::iterator c = xrdcis.begin(); c != xrdcis.end(); c++)
			//foreach(xSpringDamperConnectionInformation c, xsdcis)
		{
			xrdci[ct++] = *c;
		}


	}
	if (!connection_data)
	{
		connection_data = new xRSDAConnectionData[cnt];
		free_angle = new double[cnt];
		unsigned int ct = 0;
		for (list<xRSDAConnectionData>::iterator c = clist.begin(); c != clist.end(); c++)
			//foreach(xSpringDamperConnectionData c, clist)
			connection_data[ct++] = *c;
	}
	nkcvalue = cvalue.size();
	nrdci = xrdcis.size();
	nConnection = cnt;
}

void xRotationSpringDamperForce::ConvertGlobalToLocalOfBodyConnectionPosition(unsigned int i, xPointMass* pm)
{
	/*size_t sid = attached_body_info[i].sid;
	xSpringDamperBodyConnectionData *d = NULL;
	for (unsigned int i = 0; i < attached_body_info[i].nconnection; i++)
	{
		d = &attached_body_data[sid + i];
		vector3d loc = new_vector3d(d->rx, d->ry, d->rz);
		vector3d new_loc = pm->toLocal(loc - pm->Position());
		d->rx = new_loc.x;
		d->ry = new_loc.y;
		d->rz = new_loc.z;
	}*/
}

size_t xRotationSpringDamperForce::NumSpringDamperConnection()
{
	return nrdci;
}

size_t xRotationSpringDamperForce::NumSpringDamperConnectionList()
{

	return nConnection;
}

size_t xRotationSpringDamperForce::NumSpringDamperConnectionValue()
{
	return nkcvalue;
}

size_t xRotationSpringDamperForce::NumSpringDamperBodyConnection()
{
	return nBodyAttached;
}

size_t xRotationSpringDamperForce::NumSpringDamperBodyConnectionData()
{
	return nBodyAttachedData;
}

xRSDAConnectionInformation* xRotationSpringDamperForce::xSpringDamperConnection()
{
	return xrdci;
}

xRSDAConnectionData* xRotationSpringDamperForce::xSpringDamperConnectionList()
{
	return connection_data;
}

xSpringDamperCoefficient* xRotationSpringDamperForce::xSpringDamperCoefficientValue()
{
	return kc_value;
}

xSpringDamperBodyConnectionInfo* xRotationSpringDamperForce::xSpringDamperBodyConnectionInformation()
{
	return attached_body_info;
}

xRSDABodyAttachedData* xRotationSpringDamperForce::XSpringDamperBodyConnectionDataList()
{
	return attached_body_data;
}

double* xRotationSpringDamperForce::FreeAngle()
{
	return free_angle;
}

void xRotationSpringDamperForce::initializeAttachedPointForDEM(double* pos, double* ep)
{
	/*vector4d* p = (vector4d*)pos;
	euler_parameters* e = (euler_parameters*)ep;
	for (unsigned int i = 0; i < nsdci; i++)
	{
		unsigned int id = xsdci[i].id;
		vector3d ri = new_vector3d(p[id].x, p[id].y, p[id].z);
		vector3d rj = new_vector3d(0, 0, 0);
		for (unsigned int j = 0; j < xsdci[i].ntsda; j++)
		{
			unsigned int sid = xsdci[i].sid + j;
			xSpringDamperConnectionData *xsd = &connection_data[sid];
			rj = new_vector3d(p[xsd->jd].x, p[xsd->jd].y, p[xsd->jd].z);
			vector3d L = rj - ri;
			double l = length(L);
			xsd->init_l = l;
		}
	}
	for (unsigned int i = 0; i < nBodyConnection; i++)
	{
		xSpringDamperBodyConnectionInfo info = connection_body_info[i];
		xParticleObject* xpo = dynamic_cast<xParticleObject*>(xDynamicsManager::This()->XObject()->XObject(std::string(info.cbody)));
		unsigned int mid = xpo->MassIndex();
		for (unsigned int j = 0; j < info.nconnection; j++)
		{
			xSpringDamperBodyConnectionData *bd = &connection_body_data[info.sid + j];
			vector3d pp = new_vector3d(p[mid].x, p[mid].y, p[mid].z);
			vector3d ri = new_vector3d(p[bd->ci].x, p[bd->ci].y, p[bd->ci].z);
			vector3d rj = new_vector3d(bd->rx, bd->ry, bd->rz);
			vector3d rp = ToLocal(e[mid], rj - pp);
			bd->rx = rp.x; bd->ry = rp.y; bd->rz = rp.z;
			bd->init_l = length(ri - rj);
		}
	}*/
}

void xRotationSpringDamperForce::xCalculateForce(const xVectorD& q, const xVectorD& qd)
{
	vector4d QRi;
	vector4d QRj;
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix34d Gi = GMatrix(ei);
	matrix34d Gj = GMatrix(ej);
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	vector3d gi = Ai * g_i;
	vector3d fi = Ai * f_i;
	vector3d fj = Aj * f_j;
	int _udrl = udrl;
	bool isSin;
	theta = xUtilityFunctions::RelativeAngle(udrl, theta, n_rev, gi, fi, fj, isSin);
	double _theta = theta + 2 * n_rev * M_PI;
	double dsin = sin(_theta);
	double dcos = cos(_theta);
	if (isSin) 
		dtheta = dot(fj, BMatrix(ei, g_i) * edi) + dot(gi, BMatrix(ej, f_j) * edj) / dcos;
	else 
		dtheta = dot(fj, BMatrix(ei, f_i) * edi) + dot(fi, BMatrix(ej, f_j) * edj) / dsin;
	n = k * _theta + c * dtheta;
	QRi = 2.0 * n * (Gi * h_i);
	QRj = -2.0 * n * (Gj * h_j);
	if (i){
		i_ptr->addAxialForce(0, 0, 0);
		i_ptr->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
	}
	if (j){
		j_ptr->addAxialForce(0, 0, 0);
		j_ptr->addEulerParameterMoment(QRj.x, QRj.y, QRj.z, QRj.w);
	}
}

void xRotationSpringDamperForce::xCalculateForceBodyAndPoint(vector3d & mi, vector3d & mj)
{
	vector4d QRi;
	vector4d QRj;
	xPoint2Spring* ps = dynamic_cast<xPoint2Spring*>(g_ptr);
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = *ps->ep0;// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = i_ptr->DEulerParameters();
	euler_parameters edj = *ps->ev0;
	//euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	//euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix34d Gi = GMatrix(ei);
	matrix34d Gj = GMatrix(ej);
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	vector3d gi = Ai * g_i;
	vector3d fi = Ai * f_i;
	vector3d fj = Aj * f_j;
	int _udrl = udrl;
	bool isSin;
	theta = xUtilityFunctions::RelativeAngle(udrl, theta, n_rev, gi, fi, fj, isSin);
	double _theta = theta + 2 * n_rev * M_PI;
	double dsin = sin(_theta);
	double dcos = cos(_theta);
	if (isSin)
		dtheta = dot(fj, BMatrix(ei, g_i) * edi) + dot(gi, BMatrix(ej, f_j) * edj) / dcos;
	else
		dtheta = dot(fj, BMatrix(ei, f_i) * edi) + dot(fi, BMatrix(ej, f_j) * edj) / dsin;
	n = k * _theta + c * dtheta;
	QRi = 2.0 * n * (Gi * h_i);
	QRj = -2.0 * n * (Gj * h_j);
	//i_ptr->addAxialForce(0, 0, 0);
	//i_ptr->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
	mi = 0.5 * LMatrix(ei) * QRi;
	mj = 0.5 * LMatrix(ej) * QRj;

	//mj->addEulerParameterMoment(QRj.x, QRj.y, QRj.z, QRj.w);
}

void xRotationSpringDamperForce::xCalculateForceForDEM(
	double* pos, double* vel, double* ep, double* ev, double* ms, double* force, double* moment)
{
	vector4d* p = (vector4d*)pos;
	vector3d* v = (vector3d*)vel;
	vector3d* f = (vector3d*)force;
	vector3d* m = (vector3d*)moment;
	euler_parameters* e = (euler_parameters*)ep;
	euler_parameters* ed = (euler_parameters*)ev;
	for (unsigned int i = 0; i < nBodyAttachedData; i++) {
		xRSDABodyAttachedData *ab = &attached_body_data[i];
		xSpringDamperCoefficient kc = kc_value[ab->kc_id];
		euler_parameters ei = e[i];// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
		euler_parameters ej = e[ab->jd];// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
		euler_parameters edi = ed[i];// new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
		euler_parameters edj = ed[ab->jd];// new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
		matrix34d Gi = GMatrix(ei);
		matrix34d Gj = GMatrix(ej);
		matrix33d Ai = GlobalTransformationMatrix(ei);
		matrix33d Aj = GlobalTransformationMatrix(ej);
		vector3d _gi = new_vector3d(ab->gix, ab->giy, ab->giz);
		vector3d _gj = new_vector3d(ab->gjx, ab->gjy, ab->gjz);
		vector3d _fi = new_vector3d(ab->fix, ab->fiy, ab->fiz);// f_i;
		vector3d _fj = new_vector3d(ab->fjx, ab->fjy, ab->fjz);// f_j;
		vector3d gi = Ai * _gi;
		vector3d fi = Aj * _fi;
		vector3d fj = Aj * _fj;
		vector3d _hi = cross(_fi, _gi);
		vector3d _hj = cross(_fj, _gj);
		bool isSin;
		ab->theta = xUtilityFunctions::RelativeAngle(ab->udrl, ab->theta, ab->n_rev, gi, fi, fj, isSin);
		double _theta = ab->theta + 2.0 * n_rev * M_PI;
		double dsin = sin(_theta);
		double dcos = cos(_theta);
		if (isSin) {
			dtheta = dot(fj, BMatrix(ei, _gi) * edi) + dot(gi, BMatrix(ej, _fj) * edj) / dcos;
		}
		else {
			dtheta = dot(fj, BMatrix(ei, _fi) * edi) + dot(fi, BMatrix(ej, _fj) * edj) / dsin;
		}
		double n = k * _theta + c * dtheta;
		vector4d QRi = 2.0 * n * Gi * _hi;
		vector4d QRj = -2.0 * n * Gj * _hj;
		m[ab->jd] = 0.5 * LMatrix(ei) * QRj;

	}
	//for (unsigned int i = 0; i < nrdci; i++)
	//{
	//	unsigned int id = xrdci[i].id;
	//	vector3d ri = new_vector3d(p[id].x, p[id].y, p[id].z);
	////	vector3d vi = v[id];
	//	vector3d rj = new_vector3d(0, 0, 0);
	////	vector3d vj = new_vector3d(0, 0, 0);
	//	for (unsigned int j = 0; j < xrdci[i].nrsda; j++)
	//	{
	//		unsigned int sid = xrdci[i].sid + j;
	//		xRSDAConnectionData *xsd = &connection_data[sid];
	//		xSpringDamperCoefficient kc = kc_value[xsd->kc_id];
	//		xsd->theta = xUtilityFunctions::RelativeAngle(xsd->udrl, xsd->theta, xsd->n_rev, )
	////		rj = new_vector3d(p[xsd.jd].x, p[xsd.jd].y, p[xsd.jd].z);
	////		vj = v[xsd.jd];
	////		vector3d L = rj - ri;
	////		l = length(L);
	////		vector3d dL = vj - vi;
	////		double dl = dot(L, dL) / l;
	////		double fr = kc.k * (l - xsd.init_l) + kc.c * dl;
	////		//std::cout << "tsda_dem - " << i << " : " << xsd.jd << " & " << l << " & " << dl << " => " << fr << std::endl;
	////		vector3d Q = (fr / l) * L;

	////		f[id] += Q;
	//	}
	//};
	//for (unsigned int i = 0; i < nBodyConnection; i++)
	//{
	//	xSpringDamperBodyConnectionInfo info = connection_body_info[i];
	//	xParticleObject* xpo = dynamic_cast<xParticleObject*>(xDynamicsManager::This()->XObject()->XObject(std::string(info.cbody)));
	//	unsigned int mid = xpo->MassIndex();
	//	//xPointMass* pm = xDynamicsManager::This()->XMBDModel()->XMass(info.cbody.toStdString());
	//	vector3d ri = new_vector3d(p[mid].x, p[mid].y, p[mid].z);// pm->Position();
	//	vector3d rj = new_vector3d(0, 0, 0);
	//	vector3d vi = v[mid];// pm->Velocity();
	//	vector3d vj = new_vector3d(0, 0, 0);
	//	euler_parameters ei = e[mid];// pm->EulerParameters();
	//	euler_parameters edi = ed[mid];// pm->DEulerParameters();
	//	matrix33d Ai = GlobalTransformationMatrix(ei);
	//	size_t sid = connection_body_info[i].sid;
	//	xSpringDamperBodyConnectionData d = { 0, };

	//	for (unsigned int j = 0; j < connection_body_info[i].nconnection; j++)
	//	{
	//		d = connection_body_data[sid + j];
	//		xSpringDamperCoefficient kc = kc_value[d.kc_id];
	//		vector4d d_pos = p[d.ci];
	//		rj = new_vector3d(d_pos.x, d_pos.y, d_pos.z);
	//		vj = v[d.ci];
	//		vector3d lp = new_vector3d(d.rx, d.ry, d.rz);
	//		L = rj - ri - Ai * lp;
	//		l = length(L);
	//		vector3d dL = vj - vi - BMatrix(edi, lp) * ei;
	//		dl = dot(L, dL) / l;
	//		double fr = kc.k * (l - d.init_l) + kc.c * dl;

	//		vector3d Qi = (fr / l) * L;
	//		vector3d Qj = -Qi;
	//		vector4d QRi = (fr / l) * BMatrix(ei, lp) * L;
	//		f[mid] += Qi;// pm->addAxialForce(Qi.x, Qi.y, Qi.z);
	//		//pm->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
	//		m[mid] += 0.5 * LMatrix(ei) * QRi;
	//		f[d.ci] += Qj;
	//	}
	//}
}

void xRotationSpringDamperForce::xCalculateForceFromDEM(
	unsigned int ci, xPointMass* pm, const double* pos, const double* vel)
{
	//xSpringDamperBodyConnectionInfo info = connection_body_info[i];
	////xPointMass* pm = xDynamicsManager::This()->XMBDModel()->XMass(info.cbody.toStdString());
	//vector3d ri = pm->Position();
	//vector3d rj = new_vector3d(0, 0, 0);
	//vector3d vi = pm->Velocity();
	//vector3d vj = new_vector3d(0, 0, 0);
	//euler_parameters ei = pm->EulerParameters();
	//euler_parameters edi = pm->DEulerParameters();
	//matrix33d Ai = GlobalTransformationMatrix(ei);
	//size_t sid = connection_body_info[i].sid;
	//xSpringDamperBodyConnectionData d = { 0, };
	//vector4d* dem_pos = (vector4d*)pos;
	//vector3d* dem_vel = (vector3d*)vel;
	//for (unsigned int i = 0; i < connection_body_info[i].nconnection; i++)
	//{
	//	d = connection_body_data[sid + i];
	//	xSpringDamperCoefficient kc = kc_value[d.kc_id];
	//	vector4d d_pos = dem_pos[d.ci];
	//	rj = new_vector3d(d_pos.x, d_pos.y, d_pos.z);
	//	vj = dem_vel[d.ci];
	//	vector3d lp = new_vector3d(d.rx, d.ry, d.rz);
	//	L = rj - ri - Ai * lp;
	//	l = length(L);
	//	vector3d dL = vj - vi - BMatrix(edi, lp) * ei;
	//	dl = dot(L, dL) / l;
	//	double fr = kc.k * (l - d.init_l) + kc.c * dl;
	//	if (fr != 0)
	//		fr = fr;
	//	vector3d Qi = (fr / l) * L;
	//	vector3d Qj = -Qi;
	//	vector4d QRi = (fr / l) * BMatrix(ei, lp) * L;
	//	pm->addAxialForce(Qi.x, Qi.y, Qi.z);
	//	pm->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
	//}
}

void xRotationSpringDamperForce::xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix34d Gi = GMatrix(ei);
	matrix34d Gj = GMatrix(ej);
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix34d Big = BMatrix(ei, g_i);
	matrix34d Bif = BMatrix(ej, f_i);
	matrix34d Bjf = BMatrix(ej, f_j);
	matrix34d dBig = BMatrix(edi, g_i);
	matrix34d dBif = BMatrix(edi, f_i);
	vector3d gi = Ai * g_i;
	vector3d fi = Ai * f_i;
	vector3d fj = Aj * f_j;
	double fiTfj = dot(fi, fj);
	double giTfj = dot(gi, fj);
	vector4d n_p_i = k * (fiTfj * (fj * Big) - giTfj * (fj * Bif)) + c * ((edj * Bjf) * (fiTfj * Big - giTfj * Bif) + fj * (fiTfj * dBig) - fj * (giTfj * dBif));
	vector4d n_p_j = k * (fiTfj * (fj * Bjf) - giTfj * (fj * Bjf)) + c * (edi * ((fiTfj * Big - giTfj * Bif) * Bjf) + gi * (fiTfj * dBig) - fi * (giTfj * dBif));//;//) * (fiTfj * Big - giTfj * Bif) + fj * (fiTfj * dBig) - fj * (giTfj * dBif));
	matrix44d Qiq = (2.0 * Gi) * (h_i * n_p_i) + 2.0 * n * MMatrix(h_i);
	matrix44d Qjq = (2.0 * Gj) * (h_j * n_p_j);
	int irc = (i - 1) * xModel::OneDOF();
	int jrc = (j - 1) * xModel::OneDOF();
	if (i)
	{
		lhs.plus(irc + 3, irc + 3, -mul * Qiq);
	}
	if (j)
	{
		lhs.plus(jrc + 3, jrc + 3, -mul * Qjq);
	}
}

void xRotationSpringDamperForce::xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	vector3d vi = new_vector3d(qd(si + 0), qd(si + 1), qd(si + 2));
	vector3d vj = new_vector3d(qd(sj + 0), qd(sj + 1), qd(sj + 2));
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix33d dAi = DGlobalTransformationMatrix(ei, edi);
	matrix33d dAj = DGlobalTransformationMatrix(ej, edj);
	matrix34d Bi = BMatrix(ei, spi);
	matrix34d Bj = BMatrix(ej, spj);
	double pow2l = 1.0 / pow(l, 2.0);
	matrix33d coe = pow2l * c * L * L;// transpose(L, L) / pow(l, 2.0);
	matrix43d coeiN = pow2l * c * Bi * (L * L);// transpose(B(action->getEP(), spj), transpose(L, L) / pow(l, 2.0));
	matrix43d coejN = pow2l * c * Bj * (L * L);
	int irc = (i - 1) * xModel::OneDOF();
	int jrc = (j - 1) * xModel::OneDOF();
	if (i)
	{
		lhs.plus(irc, irc, -mul * coe);
		lhs.plus(irc, irc + 3, -mul * coe * Bi);
		lhs.plus(irc + 3, irc, -mul * coeiN);
		lhs.plus(irc + 3, irc + 3, -mul * coeiN * Bi);
	}

	if (j)
	{
		lhs.plus(jrc, jrc, -mul * coe);
		lhs.plus(jrc, jrc + 3, -mul * coe * Bj);
		lhs.plus(jrc + 3, jrc, -mul * coejN);
		lhs.plus(jrc + 3, jrc + 3, -mul * coejN * Bj);
	}
	if (i && j)
	{
		lhs.plus(irc, jrc, mul * coe);
		lhs.plus(irc, jrc + 3, mul * coe * Bj);
		lhs.plus(irc + 3, jrc, mul * coeiN);
		lhs.plus(irc + 3, jrc + 3, mul * coeiN * Bj);
		lhs.plus(jrc, irc, mul * coe);
		lhs.plus(jrc, irc + 3, mul * coe * Bi);
		lhs.plus(jrc + 3, irc, mul * coejN);
		lhs.plus(jrc + 3, irc + 3, mul * coejN * Bi);
	}
}