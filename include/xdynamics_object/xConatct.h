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
	void setContactForceModel(xContactForceModelType xcfmt);
	void setCohesion(double d);
	void setRestitution(double d);
	void setFriction(double d);
	void setStiffnessRatio(double d);
	//contactForce_type ForceMethod() const { return f_type; }
	xMaterialPair MaterialPropertyPair() const;
	xContactForceModelType ContactForceModel() const;
	device_contact_property* DeviceContactProperty() const;// { return dcp; }
	xContactPairType PairType() const;

	xContactParameters getContactParameters(
		double ir, double jr,
		double im, double jm,
		double iE, double jE,
		double ip, double jp,
		double is, double js);
	// 		contactForce_type cft, double rest, double ratio, double fric);
	void setMaterialPair(xMaterialPair _mpp);// { mpp = _mpp; }

	virtual void collision(
		double r, double m, 
		vector3d& pos, vector3d& vel, 
		vector3d& omega, vector3d& fn, vector3d& ft);
	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0)) = 0;
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
	double cohesionForce(double coh_r, double coh_e, double Fn);
	void DHSModel(xContactParameters& c, double cdist, double& ds, double& dots, vector3d& cp, vector3d& dv, vector3d& unit, vector3d& F, vector3d& M);
	bool is_enabled;
	double ignore_time;
	QString name;
	xContactPairType type;
//	contactForce_type f_type;
	xContactForceModelType force_model;
	xMaterialPair mpp;
	//contact_parameter cp;
 	device_contact_property* dcp;
// 	device_contact_property_f* dcp_f;
	xObject* iobj;
	xObject* jobj;

	double cohesion;
	double restitution;
	double stiffnessRatio;
	double friction;
};

#endif