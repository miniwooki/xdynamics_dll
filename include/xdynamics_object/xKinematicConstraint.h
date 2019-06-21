#ifndef XKINEMATICCONSTRAINT_H
#define XKINEMATICCONSTRAINT_H

#include "xdynamics_decl.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"
#include "xdynamics_object/xDrivingConstraint.h"

//typedef EXPORT_ALGEBRA_API xMatrixD matrixd;

class XDYNAMICS_API xKinematicConstraint
{
public:
	enum cType{ FIXED = 0, SPHERICAL, REVOLUTE, TRANSLATIONAL, UNIVERSAL, CABLE, GEAR, COINCIDE };
	typedef struct  
	{
		double time;
		vector3d location;
		vector3d iaforce;
		vector3d irforce;
		vector3d jaforce;
		vector3d jrforce;
	}kinematicConstraint_result;
	xKinematicConstraint();
	xKinematicConstraint(std::wstring _name, cType _type, std::wstring _i, std::wstring _j);
	virtual ~xKinematicConstraint();

	QString Name();
	cType Type();
	unsigned int NumConst();
	int IndexBaseBody();
	int IndexActionBody();
	std::wstring BaseBodyName();
	std::wstring ActionBodyName();
	xPointMass* BaseBody();
	xPointMass* ActionBody();
	void setBaseBodyIndex(int _i);
	void setActionBodyIndex(int _j);
	void AllocResultMemory(unsigned int _s);
	void setLocation(double x, double y, double z);
	void SetupDataFromStructure(xPointMass* base, xPointMass* action, xJointData& d);
	QVector<kinematicConstraint_result>* XKinematicConstraintResultPointer();
// 		double lx, double ly, double lz,
// 		double spix, double spiy, double spiz, double fix, double fiy, double fiz, double gix, double giy, double giz,
// 		double spjx, double spjy, double spjz, double fjx, double fjy, double fjz, double gjx, double gjy, double gjz);
	void ExportResults(std::fstream& of);
	virtual void ConstraintEquation(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul) = 0;
	virtual void ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr) = 0;
	virtual void DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul) = 0;
	virtual void SaveStepResult(unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr) = 0;
	virtual void GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul) = 0;

protected:
	matrix34d spherical_constraintJacobian_e(euler_parameters& e, vector3d& s);
	void spherical_differentialJacobian(xMatrixD& lhs, vector3d& L);
	vector3d spherical_constraintEquation(vector3d& ri, vector3d& rj, vector3d& si, vector3d& sj);
	vector3d spherical_gamma(euler_parameters dei, euler_parameters dej, vector3d si, vector3d sj);
	vector4d dot_1_constraintJacobian(vector3d& si, vector3d& sj, euler_parameters& e);
	double dot_1_constraintEquation(vector3d& s1, vector3d& s2);
	void dot_1_differentialJacobian(xMatrixD& lhs, vector3d& s_i, vector3d& s_j, euler_parameters& ei, euler_parameters& ej, double L);
	double dot_1_gamma(euler_parameters ei, euler_parameters ej, vector3d ai, vector3d aj, euler_parameters dei, euler_parameters dej);
	//vector4d dot_2_constraintJacobian_base(vector3d& dij, vector3d& s1_global, vector3d& s2_local, euler_parameters& e) const;
	vector4d dot_2_constraintJacobian(vector3d& s1_global, vector3d& s2_local, euler_parameters& e);
	void dot_2_differentialJacobian(xMatrixD& lhs, vector3d& a_local, vector3d& s1_local, vector3d& s2_local, vector3d& dij, euler_parameters& ei, euler_parameters& ej, double L);
	double dot_2_constraintEquation(vector3d& s1_global, vector3d& s2_global, vector3d& s1_local, vector3d& s2_local);
	double dot_2_gamma(euler_parameters ei, euler_parameters ej, vector3d aj, vector3d si, vector3d sj, vector3d dij, vector3d drj, vector3d dri, euler_parameters dei, euler_parameters dej);
	double relative_rotation_constraintGamma(double theta, euler_parameters& ei, euler_parameters& ej, euler_parameters& dei, euler_parameters& dej, vector3d& global_fi, vector3d& global_gi, vector3d& global_fj);
	double relative_translation_constraintGamma(vector3d& dij, vector3d& global_ai, vector3d& local_ai, vector3d& vi, vector3d& vj, euler_parameters& ei, euler_parameters& ej, euler_parameters& dei, euler_parameters& dej);
	vector4d relative_rotation_constraintJacobian_e_i(double theta, euler_parameters& ei, euler_parameters& ej, vector3d& global_fj);
	vector4d relative_rotation_constraintJacobian_e_j(double theta, euler_parameters& ei, euler_parameters& ej, vector3d& global_fi, vector3d& global_gi);

// 	vector3d relative_distance_constraintJacobian_r_i(euler_parameters& e);
// 	vector3d relative_distance_constraintJacobian_r_j(euler_parameters& e);
// 	vector4d relative_distance_constraintJacobian_e_i(vector3d& dist, vector3d& global_hi, euler_parameters& ei);
// 	vector4d relative_distance_constraintJacobian_e_j(vector3d& global_hi, euler_parameters& ej);
// 	double dot_1_differential(vector3d& ai, vector3d& aj, euler_parameters& pi, euler_parameters& pj, euler_parameters& dpi, euler_parameters& dpj);
// 	double dot_2_differential(vector3d& ai, vector3d& dri, vector3d& drj, vector3d& dij, euler_parameters& pi, euler_parameters& pj, euler_parameters& dpi, euler_parameters& dpj);
	QString name;
	QVector<kinematicConstraint_result> kcrs;
	unsigned int nConst;		// The number of constraint 
	unsigned int nr_part;	
	friend class xDrivingConstraint;
	int i, j;
	xPointMass *i_ptr;
	xPointMass *j_ptr;
	QString base, action;
	cType type;					// Joint type
	vector3d location;			// Joint location
	vector3d spi;				// Local position for i body
	vector3d spj;				// Local position for j body
	vector3d fi, fj;			
	vector3d hi, hj;
	vector3d gi, gj;

	//static xSparseD djaco;
};

#endif