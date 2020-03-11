#ifndef XDRIVINGCONSTRAINT_H
#define XDRIVINGCONSTRAINT_H

#include "xdynamics_decl.h"
#include "xdynamics_object/xKinematicConstraint.h"

//class xKinematicConstraint;

class XDYNAMICS_API xDrivingConstraint
{
	
public:
	enum { ROTATION_DRIVING = 0, TRANSLATION_DRIVING };
	xDrivingConstraint();
	xDrivingConstraint(std::string _name, xKinematicConstraint* _kc);
	~xDrivingConstraint();

	unsigned int RevolutionCount();
	unsigned int DerivativeRevolutionCount();
	double RotationAngle();
	double EndTime();
	void setRevolutionCount(unsigned int _n_rev);
	void setDerivativeRevolutionCount(unsigned int _dn_rev);
	void setRotationAngle(double _theta);
	void define(xVectorD& q);
	std::string Name();
	int get_driving_type();
	void setStartTime(double st);
	void setEndTime(double et);
	void setConstantVelocity(double cv);
	void setFixAfterDrivingEnd(bool b);
	void ImportResults(std::string f);
	void ExportResults(std::fstream& of);
	void ConstraintGamma(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul);
	void ConstraintEquation(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul);
	void ConstraintDerivative(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct, double mul);
	void ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr, double ct);
	void DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& q_1, double* lm, unsigned int sr, double mul, double ct);
	xKinematicConstraint::kinematicConstraint_result GetStepResult(unsigned int part,double ct, xVectorD& q, xVectorD& q_1, double* L, unsigned int sr);
	void DerivateEquation(xVectorD& v, xVectorD& q, xVectorD& qd, int sr, double ct, double mul);


private:
//	double RelativeAngle(double ct, vector3d& gi, vector3d& fi, vector3d& fj);
	int udrl;
	int d_udrl;
	xstring name;
	int type;
	bool is_fix_after_end;
	double last_ce;
	double plus_time;
	double start_time;
	double end_time;
	//int maxnnz;
	double init_v;
	double cons_v;
//	QVector<xKinematicConstraint::kinematicConstraint_result> kcrs;
	double theta;
	int n_rev;
	int dn_rev;
	unsigned int srow;
	//unsigned int nr_part;
	xKinematicConstraint* kconst;
	//Type type;
};

#endif