#include "xdynamics_object/xConatct.h"
#include "xdynamics_simulation//xSimulation.h"

xContact::xContact()
	: dcp(NULL)
	, iobj(NULL)
	, jobj(NULL)
	, cohesion(0)
	, restitution(0)
	, stiffnessRatio(0)
	, friction(0)
	, rolling_factor(0)
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

xContactParameters xContact::getContactParameters(double ir, double jr, double im, double jm, double iE, double jE, double ip, double jp, double is, double js)
{
	xContactParameters cp;
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	cp.coh_e = ((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE);
	double lne = log(restitution);
	double beta = 0.0;
// 	switch (f_type)
// 	{
// 	case DHS:
	beta = (M_PI / log(restitution));
	cp.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	cp.vn = sqrt((4.0 * Meq * cp.kn) / (1.0 + beta * beta));
	cp.ks = cp.kn * stiffnessRatio;
	cp.vs = cp.vn * stiffnessRatio;
//		break;
	//}
	cp.coh_r = Req;
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

double xContact::cohesionForce(double coh_r, double coh_e, double Fn)
{
	double cf = 0.0;
	if (cohesion){
		double rcp = (3.0 * coh_r * (-Fn)) / (4.0 * (1.0 / coh_e));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = cohesion * Ac;
	}
	return cf;
}

void xContact::DHSModel(
	xContactParameters& c, double cdist, double& ds, double& dots, 
	vector3d& cp, vector3d& dv, vector3d& unit, vector3d& Fn, vector3d& Ft, vector3d& M)
{
	//vector3d Fn, Ft;
	double fsn = (-c.kn * pow(cdist, 1.5));
	double fca = cohesionForce(c.coh_r, c.coh_e, fsn);
	double fsd = c.vn * dot(dv, unit);
	Fn = (fsn + fca + fsd) * unit;
	vector3d e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e){
		vector3d s_hat = e / mag_e;
		double s_dot = dot(dv, s_hat);
		ds = ds + xSimulation::dt * (s_dot + dots);
		dots = s_dot;
		//double ds = mag_e * xSimulation::dt;
		double ft1 = c.ks * ds + c.vs * dot(dv, s_hat);
		double ft2 = friction * length(Fn);
		Ft = min(ft1, ft2) * s_hat;
		M = cross(cp, Ft);
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
		restitution, friction, 0.0, cohesion, stiffnessRatio, rolling_factor
	};
	checkCudaErrors(cudaMalloc((void**)&dcp, sizeof(device_contact_property)));
	checkCudaErrors(cudaMemcpy(dcp, &hcp, sizeof(device_contact_property), cudaMemcpyHostToDevice));
}

void xContact::collision(double r, double m, vector3d& pos, vector3d& vel, vector3d& omega, vector3d& fn, vector3d& ft)
{

}