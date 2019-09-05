#ifndef XCONTACT_H
#define XCONTACT_H

#include "xdynamics_decl.h"
#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_object/xContactPairList.h"
#include <QtCore/QString>

class xObject;

class XDYNAMICS_API xContact
{
public:
	xContact();
	xContact(std::string _name, xContactPairType xcpt);
	xContact(const xContact& xc);
	virtual ~xContact();

// 	double IgnoreTime() { return ignore_time; }
// 	void setIgnoreTime(double _t) { ignore_time = _t; }
	bool IsEnabled();
	void setEnabled(bool b);
	QString Name() const;
	xObject* FirstObject() const;
	xObject* SecondObject() const;
	void setFirstObject(xObject* o1);
	void setSecondObject(xObject* o2);
	//void setContactParameters(double r, double rt, double f, double c);
	double Cohesion() const;
	double Restitution() const;
	double Friction() const;
	double StiffnessRatio() const;
	double RollingFactor() const;
	double StiffMultiplyer() const;
	static void setContactForceModel(xContactForceModelType xcfmt);
	void setCohesion(double d);
	void setRestitution(double d);
	void setFriction(double d);
	void setStiffnessRatio(double d);
	void setRollingFactor(double d);
	void setStiffMultiplyer(double d);
	static double3* deviceBodyForce();
	static double3* deviceBodyMoment();
	//contactForce_type ForceMethod() const { return f_type; }
	xMaterialPair MaterialPropertyPair() const;
	static xContactForceModelType ContactForceModel();
	device_contact_property* DeviceContactProperty() const;// { return dcp; }
	xContactPairType PairType() const;

	xContactParameters getContactParameters(
		double ir, double jr,
		double im, double jm,
		double iE, double jE,
		double ip, double jp,
		double is, double js,
		double rest, double ratio,
		double fric, double rfric, double coh);
	// 		contactForce_type cft, double rest, double ratio, double fric);
	void setMaterialPair(xMaterialPair _mpp);// { mpp = _mpp; }
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np) = 0;

	virtual void cudaMemoryAlloc(unsigned int np);
	static unsigned int count;

protected:
	
	double cohesionSeperationDepth(double coh, double ir, double jr, double ip, double jp, double Ei, double Ej);
	double cohesionForce(double coh, double cdist, double coh_r, double coh_e, double coh_s, double Fn);
	double JKRSeperationForce(xContactParameters& c, double coh);
	void DHSModel(xContactParameters& c, double cdist, double& ds, 
		double& dots, double coh, vector3d& dv, vector3d& unit, vector3d& Fn, vector3d& Ft);
	void Hertz_Mindlin(xContactParameters& c, double cdist, double& ds,
		double& dots, double coh, vector3d& dv, vector3d& unit, vector3d& Fn, vector3d& Ft);
	void RollingResistanceForce(
		double rf, double ir, double jr, vector3d rc,
		vector3d Fn, vector3d Ft, double& Mr, vector3d& Tmax);
	bool is_enabled;
	double ignore_time;
	QString name;
	xContactPairType type;
//	contactForce_type f_type;
	static xContactForceModelType force_model;
	xMaterialPair mpp;
	//contact_parameter cp;
 	device_contact_property* dcp;
// 	device_contact_property_f* dcp_f;
	xObject* iobj;
	xObject* jobj;
	static double3* db_force;
	static double3* db_moment;

	double cohesion;
	double restitution;
	double stiffnessRatio;
	double friction;
	double rolling_factor;
	double stiff_multiplyer;
};


#endif