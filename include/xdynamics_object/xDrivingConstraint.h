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

	void define(xVectorD& q);
	std::string Name();
	void setStartTime(double st);
	void setConstantVelocity(double cv);
	void ImportResults(std::string f);
	void ExportResults(std::fstream& of);
	//unsigned int startRow() { return srow; }
	//	unsigned int startColumn() { return scol; }
	//bool use(int i) { return use_p[i]; }
	//int maxNNZ() { return maxnnz; }
	//pointMass* ActionBody(){ return m; }
	//void updateInitialCondition();
	//double constraintEquation(double ct);
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
	double plus_time;
	double start_time;
	//int maxnnz;
	double init_v;
	double cons_v;
//	QVector<xKinematicConstraint::kinematicConstraint_result> kcrs;
	double theta;
	//unsigned int srow;
	//unsigned int scol;
	//QString name;

	unsigned int n;
	unsigned int n_rev;
	unsigned int dn_rev;
	unsigned int srow;
	unsigned int nr_part;
	xKinematicConstraint* kconst;
	//Type type;
};

#endif