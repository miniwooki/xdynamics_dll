#include "xdynamics_object/xConatct.h"
#include "xdynamics_simulation//xSimulation.h"

double* xContact::dbfx = NULL;
double* xContact::dbfy = NULL;
double* xContact::dbfz = NULL;
double* xContact::dbmx = NULL;
double* xContact::dbmy = NULL;
double* xContact::dbmz = NULL;
xContactForceModelType xContact::force_model = NO_DEFINE_CONTACT_MODEL;

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
	: name(_name)
	, type(xcpt)
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
	if (dbfx) checkCudaErrors(cudaFree(dbfx)); dbfx = NULL;
	if (dbfy) checkCudaErrors(cudaFree(dbfy)); dbfy = NULL;
	if (dbfz) checkCudaErrors(cudaFree(dbfz)); dbfz = NULL;
	if (dbmx) checkCudaErrors(cudaFree(dbmx)); dbmx = NULL;
	if (dbmy) checkCudaErrors(cudaFree(dbmy)); dbmy = NULL;
	if (dbmz) checkCudaErrors(cudaFree(dbmz)); dbmz = NULL;
}

xContactForceModelType xContact::ContactForceModel()
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

void xContact::setStaticFriction(double d)
{
	s_friction = d;
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
	double rest, double ratio, double s_fric,
	double fric, double rfric, double coh)
{
	xContactParameters cp;
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	double Seq = (2.0 * (2.0 - ip) * (1.0 + ip) / iE) + (2.0 * (2.0 - jp) * (1.0 + jp) / jE);
	Seq = 1.0 / Seq;
	cp.coh_e = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));
	double lne = log(rest);
	double beta = 0.0;
	beta = -lne * sqrt(1.0 / (lne*lne + M_PI * M_PI));
	cp.eq_e = Eeq;
	cp.eq_m = Meq;
	cp.eq_r = Req;
	cp.eq_s = Seq;
	cp.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	cp.vn = beta; //(//sqrt((4.0 * Meq * cp.kn) / (1.0 + beta * beta));
	cp.ks = force_model == HERTZ_MINDLIN_NO_SLIP ? 8.0 * Seq : cp.kn * ratio;
	cp.vs = force_model == HERTZ_MINDLIN_NO_SLIP ? 2.0 * sqrt(5.0 / 6.0) * beta : cp.vn * ratio;
	cp.s_fric = s_fric;
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
	return cf;
}

double xContact::cohesionSeperationDepth(double coh, double ir, double jr, double ip, double jp, double iE, double jE)
{
	double coh_r = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp * jp) + jE * (1 - ip * ip));
	double coh_e = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));
	double c1 = (M_PI * M_PI * coh * coh * coh_r) / (coh_e * coh_e);
	double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);
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
		cf = - sqrt(8.0 * M_PI * coh * coh_e * a3);
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

void xContact::Hertz_Mindlin(
	xContactParameters& c, double cdist, double& ds, double& dots, double coh,
	vector3d& dv, vector3d& unit, vector3d& Fn, vector3d& Ft/*, vector3d& M*/)
{
	//vector3d Fn, Ft;
	double fsn = (-c.kn * pow(cdist, 1.5));
	double fsd = c.vn * (2.0 * sqrt(c.eq_m * c.kn)) * dot(dv, unit);
	double fco = -cohesionForce(coh, cdist, c.coh_r, c.coh_e, c.coh_s, fsn + fsd);

	Fn = (fsn + fsd + fco) * unit;
	vector3d e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e) {
		vector3d s_hat = e / mag_e;
		double s_dot = dot(dv, s_hat);
		ds = ds + xSimulation::dt * (s_dot + dots);
		dots = s_dot;
		//double ds = mag_e * xSimulation::dt;
		double S_t = c.ks * sqrt(c.eq_r * cdist);
		double ft1 = S_t * ds + c.vs * sqrt(S_t * c.eq_m) * dot(dv, s_hat);
		double ft2 = c.fric * length(Fn);
		double ft3 = c.s_fric * length(Fn);
		Ft = (ft1 < ft3 ? 0.0 : ft2)/*min(ft1, ft2)*/ * s_hat;
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
xstring xContact::Name() const { return name; }
xObject* xContact::FirstObject() const { return iobj; }
xObject* xContact::SecondObject() const { return jobj; }
double xContact::Cohesion() const { return cohesion; }
double xContact::Restitution() const { return restitution; }
double xContact::Friction() const { return friction; }
double xContact::StaticFriction() const
{
	return s_friction;
}
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
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(device_contact_property), cudaMemcpyHostToDevice));
	if (!dbfx) checkCudaErrors(cudaMalloc((void**)&dbfx, sizeof(double) * np));
	if (!dbfy) checkCudaErrors(cudaMalloc((void**)&dbfy, sizeof(double) * np));
	if (!dbfz) checkCudaErrors(cudaMalloc((void**)&dbfz, sizeof(double) * np));
	if (!dbmx) checkCudaErrors(cudaMalloc((void**)&dbmx, sizeof(double) * np));
	if (!dbmy) checkCudaErrors(cudaMalloc((void**)&dbmy, sizeof(double) * np));
	if (!dbmz) checkCudaErrors(cudaMalloc((void**)&dbmz, sizeof(double) * np));
}

double xContact::StiffMultiplyer() const
{
	return stiff_multiplyer;
}

void xContact::setStiffMultiplyer(double d)
{
	stiff_multiplyer = d;
}

double* xContact::deviceBodyForceX() { return dbfx; }
double* xContact::deviceBodyForceY() { return dbfy; }
double* xContact::deviceBodyForceZ() { return dbfz; }

double * xContact::deviceBodyMomentX() { return dbmx; }
double * xContact::deviceBodyMomentY() { return dbmy; }
double * xContact::deviceBodyMomentZ() { return dbmz; }
