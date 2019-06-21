#include "xdynamics_object/xUniversalConstraint.h"

xUniversalConstraint::xUniversalConstraint()
	: xKinematicConstraint()
{

}

xUniversalConstraint::xUniversalConstraint(std::wstring _name, std::wstring _i, std::wstring _j)
	: xKinematicConstraint(_name, xKinematicConstraint::UNIVERSAL, _i, _j)
{
	//hj = new_vector3d(-0.97014250014533202344536934398782, 0, 0.24253562503633300586134233599695);
}

xUniversalConstraint::~xUniversalConstraint()
{

}

void xUniversalConstraint::ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul)
{
	// 	unsigned int idx = (i - 1) * xModel::OneDOF();
	// 	unsigned int jdx = (j - 1) * xModel::OneDOF();
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
// 	vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
// 	vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
// 	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
// 	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	vector3d v3 = mul * spherical_constraintEquation(i_ptr->Position(), j_ptr->Position(), Ai * spi, Aj * spj);
	//v3 = (rj + Aj * spj) - (ri + Ai * spi);
	ce(sr + 0) = v3.x;
	ce(sr + 1) = v3.y;
	ce(sr + 2) = v3.z;
	v3 = Aj * hj;
	double n = sqrt(dot(v3, v3));
	ce(sr + 3) = mul * dot_1_constraintEquation(v3, Ai * hi);//dot(v3, Ai * fi);
}

void xUniversalConstraint::ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	if (i)
	{
		sc = (i - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = -1;
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(ei, -spi));
		lhs.insert(sr + 3, sc + 3, dot_1_constraintJacobian(Aj * hj, hi, ei));// *BMatrix(e, fi));
		//s = gi; lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(A * hj, s, e));// *BMatrix(e, gi));
	}
	if (j)
	{
		sc = (j - 1) * xModel::OneDOF();
		lhs(sr + 0, sc + 0) = lhs(sr + 1, sc + 1) = lhs(sr + 2, sc + 2) = 1;
		lhs.insert(sr + 0, sc + 3, spherical_constraintJacobian_e(ej, spj));
		lhs.insert(sr + 3, sc + 3, dot_1_constraintJacobian(Ai * hi, hj, ej));// *BMatrix(e, hj));
		//lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(A * gi, hj, e));// *BMatrix(e, hj));
	}
}

void xUniversalConstraint::DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul)
{
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));

	vector3d L = new_vector3d(lm[0], lm[1], lm[2]);
	spherical_differentialJacobian(lhs, mul * L);
	dot_1_differentialJacobian(lhs, hi, hj, ei, ej, mul * lm[3]);
	//dot_1_differentialJacobian(lhs, gi, hj, ei, ej, mul * lm[4]);
}  

void xUniversalConstraint::SaveStepResult(
	unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int si = i * xModel::OneDOF();
// 	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// (q(si + 0), q(si + 1), q(si + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	kinematicConstraint_result kcr = { 0, };
	double* lm = L + sr;
	vector3d L3 = new_vector3d(lm[0], lm[1], lm[2]);
	vector4d rf;
	matrix34d sce;
	vector4d d1c;
	if (i)
	{
		kcr.iaforce = L3;
		sce = spherical_constraintJacobian_e(ei, spi);
		d1c = -dot_1_constraintJacobian(Aj * hj, hi, ei);
		rf = sce * L3 + lm[3] * d1c;
		kcr.irforce = 0.5 * LMatrix(ei) * rf;
	}
	if (j)
	{
		kcr.jaforce = -L3;
		sce = spherical_constraintJacobian_e(ej, -spj);
		d1c = -dot_1_constraintJacobian(Ai * hi, hj, ej);
		rf = sce * L3 + lm[3] * d1c;
		kcr.jrforce = 0.5 * LMatrix(ej) * rf;
	}
	kcr.time = ct;
	kcr.location = ri + Ai * spi;
	kcrs.push_back(kcr);
	nr_part++;
}

void xUniversalConstraint::GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters dei = i_ptr->DEulerParameters();// new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters dej = j_ptr->DEulerParameters();// ne
	vector3d v3 = mul * spherical_gamma(dei, dej, spi, spj);
	rhs(sr + 0) = v3.x;
	rhs(sr + 1) = v3.y;
	rhs(sr + 2) = v3.z;
	rhs(sr + 3) = mul * dot_1_gamma(ei, ej, hi, hj, dei, dej);
}

