#ifndef XPOINTMASS_H
#define XPOINTMASS_H

//#include "xdynamics_decl.h"
#include "xdynamics_object/xObject.h"
#include <QtCore/QTextStream>
#include <QtCore/QVector>

class XDYNAMICS_API xPointMass : public xObject
{
#define INITIAL_BUFFER_SIZE 14
#define INITIAL_ZERO_BUFFER_SIZE 40
public:
	typedef struct  
	{
		double time;
		vector3d pos, vel, acc, omega, alpha, af, am, cf, cm, hf, hm;
		euler_parameters ep, ev, ea;
	}pointmass_result;

	xPointMass(xShapeType _s = NO_SHAPE);
	xPointMass(const xPointMass& xpm);
	xPointMass(std::string _name, xShapeType _s = NO_SHAPE);
	~xPointMass();

	// Declaration set functions
	void setXpmIndex(int idx);
	void setMass(double _mass);
	void setPosition(double x, double y, double z);
	void setVelocity(double x, double y, double z);
	void setAcceleration(double x, double y, double z);
	void setEulerParameters(double e0, double e1, double e2, double e3);
	void setDEulerParameters(double e0, double e1, double e2, double e3);
	void setDDEulerParameters(double e0, double e1, double e2, double e3);
	void setSymetricInertia(double xy, double xz, double yz);
	void setDiagonalInertia(double xx, double yy, double zz);
	void setAngularVelocity(double x, double y, double z);
	void setAngularAcceleration(double x, double y, double z);
	void setAxialForce(double x, double y, double z);
	void setAxialMoment(double x, double y, double z);
	void setContactForce(double x, double y, double z);
	void setContactMoment(double x, double y, double z);
	void setHydroForce(double x, double y, double z);
	void setHydroMoment(double x, double y, double z);
	void setEulerParameterMoment(double m0, double m1, double m2, double m3);

	void addContactForce(double x, double y, double z);
	void addContactMoment(double x, double y, double z);
	void addHydroForce(double x, double y, double z);
	void addHydroMoment(double x, double y, double z);
	void addAxialForce(double x, double y, double z);
	void addAxialMoment(double x, double y, double z);
	void addEulerParameterMoment(double m0, double m1, double m2, double m3);

	// Declaration get functions
	int xpmIndex() const;
	double Mass() const;
	matrix33d Inertia() const;
	vector3d Position() const;
	vector3d Velocity() const;
	vector3d Acceleration() const;
	euler_parameters EulerParameters() const;
	euler_parameters DEulerParameters() const;
	euler_parameters DDEulerParameters() const;
	vector3d SymetricInertia() const;
	vector3d DiaginalInertia() const;
	vector3d AngularVelocity() const;
	vector3d AngularAcceleration() const;
	vector3d AxialForce() const;
	vector3d AxialMoment() const;
	vector3d ContactForce() const;
	vector3d ContactMoment() const;
	vector3d HydroForce() const;
	vector3d HydroMoment() const;
	vector4d EulerParameterMoment() const;

	QVector<pointmass_result>* XPointMassResultPointer();

	// Declaration operate functions
	void setupTransformationMatrix();
	void setupInertiaMatrix();
	matrix33d TransformationMatrix() const;

	vector3d toLocal(const vector3d& v);
	vector3d toGlobal(const vector3d& v);

	void AllocResultMomory(unsigned int _s);
	void setZeroAllForce();
	void SaveStepResult(unsigned int part, double time, xVectorD& q, xVectorD& qd, xVectorD& qdd);
	void ExportResults(std::fstream& of);
	void SetDataFromStructure(int id, xPointMassData& d);
	void ImportInitialData();
	void ExportInitialData();
	void setNewData(xVectorD& q, xVectorD& qd);
	void setNewPositionData(xVectorD& q);
	void setNewVelocityData(xVectorD& qd);
//	static void ExportResult2ASCII(std::ifstream& ifs);

protected:
	int index;
	unsigned int nr_part;
	double mass;			// mass
	vector3d syme_inertia;	// symetric inertia
	vector3d diag_inertia;  // diagonal inertia

	matrix33d inertia;		// inertia

	vector3d pos;			// position
	vector3d vel;			// velocity
	euler_parameters ep;	// euler parameters
	euler_parameters ev;	// first derivative of euler parameters
	
	vector3d af;			// axial force
	vector3d am;			// axial moment
	vector3d cf;			// contact force
	vector3d cm;			// contact moment
	vector3d hf;			// hydro force
	vector3d hm;			// hydro moment
	vector4d em;

	vector3d omega;			// angular velocity
	vector3d alpha;			// angular acceleration

	vector3d acc;			// acceleration
	euler_parameters ea;	// second derivative of euler parameters

	matrix33d A;			// transformation matrix

	QVector<pointmass_result> pmrs;

private:
	double* initial_data;
};

#endif