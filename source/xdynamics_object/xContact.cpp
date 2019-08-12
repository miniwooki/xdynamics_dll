#include "xdynamics_object/xConatct.h"
#include "xdynamics_simulation//xSimulation.h"

double3* xContact::db_force = NULL;
double3* xContact::db_moment = NULL;

xContact::xContact()
	: dcp(NULL)
	, iobj(NULL)
	, jobj(NULL)
	, cohesion(0)
	, restitution(0)
	, stiffnessRatio(0)
	, friction(0)
	, rolling_factor(0)
	, stiff_multiplyer(1.0)
{

}

xContact::xContact(std::string _name, xContactPairType xcpt)
	: name(QString::fromStdString(_name))
	, type(xcpt)
	, force_model(DHS)
	, dcp(NULL)
	, iobj(NULL)
	, jobj(NULL)
	, cohesion(0)
	, restitution(0)
	, stiffnessRatio(0)
	, friction(0)
	, rolling_factor(0)
	, stiff_multiplyer(1.0)
{

}

xContact::xContact(const xContact& xc)
	: name(xc.Name())
	, type(xc.PairType())
	, force_model(xc.ContactForceModel())
	, dcp(NULL)
	, iobj(xc.FirstObject())
	, jobj(xc.SecondObject())
	, cohesion(xc.Cohesion())
	, restitution(xc.Restitution())
	, stiffnessRatio(xc.StiffnessRatio())
	, friction(xc.Friction())
	, mpp(xc.MaterialPropertyPair())
	, rolling_factor(xc.RollingFactor())
{
	if (xc.DeviceContactProperty())
	{
		checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
		checkCudaErrors(cudaMemcpy(dcp, xc.DeviceContactProperty(), sizeof(device_contact_property), cudaMemcpyDeviceToDevice));
	}
}

xContact::~xContact()
{
	if (db_force) checkCudaErrors(cudaFree(db_force)); db_force = NULL;
	if (db_moment) checkCudaErrors(cudaFree(db_moment)); db_moment = NULL;
}

xContactForceModelType xContact::ContactForceModel() const
{
	return force_model;
}

device_contact_property* xContact::DeviceContactProperty() const
{ 
	return dcp; 
}

void xContact::setContactForceModel(xContactForceModelType xcfmt)
{
	force_model = xcfmt;
}


void xContact::setCohesion(double d)
{
	cohesion = d;
}

void xContact::setRestitution(double d)
{
	restitution = d;
}

void xContact::setFriction(double d)
{
	friction = d;
}

void xContact::setStiffnessRatio(double d)
{
	stiffnessRatio = d;
}

void xContact::setFirstObject(xObject* o1)
{
	iobj = o1;
}

void xContact::setSecondObject(xObject* o2)
{
	jobj = o2;
}

double xContact::RollingFactor() const
{
	return rolling_factor;
}

xContactParameters xContact::getContactParameters(
	double ir, double jr, 
	double im, double jm, 
	double iE, double jE, 
	double ip, double jp, 
	double is, double js,
	double rest, double ratio,
	double fric, double rfric, double coh)
{
	xContactParameters cp;
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	cp.coh_e = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));
	double lne = log(rest);
	double beta = 0.0;
// 	switch (f_type)
// 	{
// 	case DHS:
	beta = (M_PI / log(rest));
	cp.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	cp.vn = sqrt((4.0 * Meq * cp.kn) / (1.0 + beta * beta));
	cp.ks = cp.kn * ratio;
	cp.vs = cp.vn * ratio;
	cp.fric = fric;
	cp.rfric = rfric;
	cp.coh_r = Req;
	double c1 = (M_PI * M_PI * coh * coh * cp.coh_r) / (cp.coh_e * cp.coh_e);
	double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);
	cp.coh_s = gs;
	return cp;
}

void xContact::setMaterialPair(xMaterialPair _mpp)
{
	mpp = _mpp;
}

void xContact::setRollingFactor(double d)
{
	rolling_factor = d;
}

double xContact::JKRSeperationForce(xContactParameters& c, double coh)
{
	double cf = -(3.0 / 2.0) * M_PI * coh * c.coh_r;
	return cf;// Fn += -cf * u;
}

//void xContact::collision(
//	xPairData* d, double ri, double rj,
//	double mi, double mj,
//	vector3d& pi, vector3d& pj,
//	vector3d& vi, vector3d& vj,
//	vector3d& oi, vector3d& oj,
//	vector3d& F, vector3d& M,
//	double& res, vector3d& tmax,
//	xContactMaterialParameters& cmp,
//	xMaterialPair* xmps)
//{
//	vector3d m_fn = new_vector3d(0.0, 0.0, 0.0);
//	vector3d m_m = new_vector3d(0.0, 0.0, 0.0);
//	vector3d m_ft = new_vector3d(0.0, 0.0, 0.0);
//	double rcon = ri - 0.5 * d->gab;
//	vector3d u = new_vector3d(d->nx, d->ny, d->nz);
//	vector3d cpt = new_vector3d(d->cpx, d->cpy, d->cpz);
//	vector3d dcpr = cpt - pi;
//	vector3d dcpr_j = cpt - pj;
//	//vector3d cp = r * u;
//	vector3d dv = vj + cross(oj, dcpr_j) - (vi + cross(oi, dcpr));
//	//unsigned int jjjj = d->id;
//	//xContactMaterialParameters cmp = hcmp[d->id];
//	xContactParameters c = getContactParameters(
//		ri, rj,
//		mi, mj,
//		xmps[d->id].Ei, xmps[d->id].Ej,
//		xmps[d->id].Pri, xmps[d->id].Prj,
//		xmps[d->id].Gi, xmps[d->id].Gj,
//		cmp.restitution, cmp.stiffness_ratio,
//		cmp.friction, cmp.rolling_friction, cmp.cohesion);
//	if (d->gab < 0 && abs(d->gab) < abs(c.coh_s))
//	{
//		double f = JKRSeperationForce(c, cmp.cohesion);
//		double cf = cohesionForce(cmp.cohesion, d->gab, c.coh_r, c.coh_e, c.coh_s, f);
//		F -= cf * u;
//		return;//continue;
//	}
//	else if (d->isc && d->gab < 0 && abs(d->gab) > abs(c.coh_s))
//	{
//		d->isc = false;
//		return;
//	}
//	switch (force_model)
//	{
//	case DHS: DHSModel(c, d->gab, d->delta_s, d->dot_s, cmp.cohesion, dv, u, m_fn, m_ft); break;
//	}
//	/*if (cmp.cohesion && c.coh_s > d->gab)
//		d->isc = false;*/
//	RollingResistanceForce(c.rfric, ri, rj, dcpr, m_fn, m_ft, res, tmax);
//	F += m_fn + m_ft;
//	M += cross(dcpr, m_fn + m_ft);
//}

double xContact::cohesionSeperationDepth(double coh, double ir, double jr, double ip, double jp, double iE, double jE)
{
	double coh_r = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp * jp) + jE * (1 - ip * ip));
	double coh_e = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));
	double c1 = (M_PI * M_PI * coh * coh * coh_r) / (coh_e * coh_e);
	double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);
	//cp.coh_s = gs;
	return gs;
}

double xContact::cohesionForce(double coh, double cdist, double coh_r, double coh_e, double coh_s, double Fn)
{
	double cf = 0.0;
	if (coh){
		double c0 = 3.0 * coh * M_PI * coh_r;
		double ac = (9.0 * coh_r * coh_r * coh * M_PI) / (4.0 * coh_e);
		double eq = 2.0 * c0 * Fn + c0 * c0;
		if (eq <= 0)
			Fn = -0.5 * c0;
		double a3 = (3.0 * coh_r) * (Fn + c0 + sqrt(2.0 * c0 * Fn + c0 * c0)) / (4.0 * coh_e);
		//double a3 = (3.0 * coh_r) * c0 / (4.0 * coh_e);
		/*double c1 = (M_PI * M_PI * coh * coh * coh_r) / (coh_e * coh_e);
		double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);*/
	/*	if (coh_s < cdist)
		{
			cf = -(3.0 / 2.0) * M_PI * coh * coh_r;
		}
		else
		{*/
			cf = /*(4.0 * coh_e * a3) / (3.0 * coh_r)*/ - sqrt(8.0 * M_PI * coh * coh_e * a3);
		//}
			
		/*double rcp = (3.0 * coh_r * (-Fn)) / (4.0 * (1.0 / coh_e));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = coh * Ac;*/
	}
	return cf;
}

void xContact::DHSModel(
	xContactParameters& c, double cdist, double& ds, double& dots, double coh,
	vector3d& dv, vector3d& unit, vector3d& Fn, vector3d& Ft/*, vector3d& M*/)
{
	//vector3d Fn, Ft;
	double fsn = (-c.kn * pow(cdist, 1.5));
	double fsd = c.vn * dot(dv, unit);
	double fco = -cohesionForce(coh, cdist, c.coh_r, c.coh_e, c.coh_s, fsn + fsd);
	//double sum_f = fsn + fsd;
	////double fsd = 0.0;
	//if (coh)
	//{
	//	sum_f = cohesionForce(coh, cdist, c.coh_r, c.coh_e, c.coh_s, sum_f);
	//}
	
	Fn = (fsn + fsd + fco) * unit;
	vector3d e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e){
		vector3d s_hat = e / mag_e;
		double s_dot = dot(dv, s_hat);
		ds = ds + xSimulation::dt * (s_dot + dots);
		dots = s_dot;
		//double ds = mag_e * xSimulation::dt;
		double ft1 = c.ks * ds + c.vs * dot(dv, s_hat);
		double ft2 = c.fric * length(Fn);
		Ft = min(ft1, ft2) * s_hat;
		//M = cross(cp, Ft);
	}
	//F = Fn + Ft;
}

void xContact::RollingResistanceForce(
	double rf, double ir, double jr, vector3d rc,
	vector3d Fn, vector3d Ft, double& Mr, vector3d& Tmax)
{
	Tmax += cross(rc, Fn + Ft);
	double Rij = jr ? ir * jr / (ir + jr) : ir;
	Mr += Rij * rf * length(Fn);
}

bool xContact::IsEnabled() { return is_enabled; }
void xContact::setEnabled(bool b) { is_enabled = b; }
QString xContact::Name() const { return name; }
xObject* xContact::FirstObject() const { return iobj; }
xObject* xContact::SecondObject() const { return jobj; }
double xContact::Cohesion() const { return cohesion; }
double xContact::Restitution() const { return restitution; }
double xContact::Friction() const { return friction; }
double xContact::StiffnessRatio() const { return stiffnessRatio; }
//contactForce_type ForceMethod() const { return f_type; }
xMaterialPair xContact::MaterialPropertyPair() const { return mpp; }
//device_contact_property* DeviceContactProperty() const { return dcp; }
xContactPairType xContact::PairType() const { return type; }

void xContact::cudaMemoryAlloc(unsigned int np)
{
	if (dcp) return;
	device_contact_property hcp = device_contact_property
	{
		mpp.Ei, mpp.Ej, mpp.Pri, mpp.Prj, mpp.Gi, mpp.Gj,
		restitution, friction, rolling_factor, cohesion, stiffnessRatio, stiff_multiplyer
	};
	/*if (cohesion)
	{
		double Meq = jm ? (im * jm) / (im + jm) : im;
		double Req = jr ? (ir * jr) / (ir + jr) : ir;
		double Eeq = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));

		double c1 = (M_PI * M_PI * coh * coh * cp.coh_r) / (cp.coh_e * cp.coh_e);
		double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);
		cp.coh_s = gs;
	}*/
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(device_contact_property), cudaMemcpyHostToDevice));
	if (!db_force)
		checkCudaErrors(cudaMalloc((void**)&db_force, sizeof(double3) * np));
	if (!db_moment)
		checkCudaErrors(cudaMalloc((void**)&db_moment, sizeof(double3) * np));
}

//void xContact::collision(
//	xContactPairList* pairs, unsigned int i, double ri, double rj,
//	double mi, double mj, vector3d& pos, vector3d& vel, vector3d& omega,
//	double &res, vector3d& tmax, vector3d& F, vector3d& M,
//	unsigned int nco, xClusterInformation* xci, vector4d* cpos)
//{
//	unsigned int ci = 0;
//	unsigned int neach = 1;
//	vector3d cp = pos;
//	if (nco && cpos)
//	{
//		for (unsigned int j = 0; j < nco; j++)
//			if (i >= xci[j].sid && i < xci[j].sid + xci[j].count * xci[j].neach)
//				neach = xci[j].neach;
//		//ck = j / neach;
//		ci = i / neach;
//		cp = new_vector3d(cpos[ci].x, cpos[ci].y, cpos[ci].z);
//	}
//	foreach(xPairData* d, pairs->PlanePair())
//	{
//
//		
//	}
//}

double xContact::StiffMultiplyer() const
{
	return stiff_multiplyer;
}

void xContact::setStiffMultiplyer(double d)
{
	stiff_multiplyer = d;
}

double3 * xContact::deviceBodyForce()
{
	return db_force;
}

double3 * xContact::deviceBodyMoment()
{
	return db_moment;
}
