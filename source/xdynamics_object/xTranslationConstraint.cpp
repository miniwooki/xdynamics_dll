#include "xdynamics_object/xTranslationConstraint.h"
#include "xdynamics_manager/xModel.h"
#include "xdynamics_object/xPointMass.h"

xTranslationConstraint::xTranslationConstraint()
	: xKinematicConstraint()
{

}

xTranslationConstraint::xTranslationConstraint(std::string _name, std::string _i, std::string _j)
	: xKinematicConstraint(_name, xKinematicConstraint::TRANSLATIONAL, _i, _j)
{

}

xTranslationConstraint::~xTranslationConstraint()
{

}

void xTranslationConstraint::ConstraintEquation(xVectorD& ce, xVectorD& q, xVectorD& dq, unsigned int sr, double mul)
{
	//unsigned int si = i * xModel::OneDOF();
//	unsigned int sj = j * xModel::OneDOF();
	//vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	//vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	//euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	//euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	//matrix33d Ai = GlobalTransformationMatrix(ei);
	//matrix33d Aj = GlobalTransformationMatrix(ej);
	vector3d v3 = j_ptr->toGlobal(hj);// Aj * hj;
	vector3d v3f = i_ptr->toGlobal(fi);// Ai * fi;
	vector3d v3g = i_ptr->toGlobal(gi);// Ai * gi;
	ce(sr + 0) = mul * dot_1_constraintEquation(v3, v3f);
	ce(sr + 1) = mul * dot_1_constraintEquation(v3, v3g);
	v3 = j_ptr->Position() + j_ptr->toGlobal(spj) - i_ptr->Position();// rj + Aj * spj - ri;
	ce(sr + 2) = mul * dot_2_constraintEquation(v3, v3f, spi, fi);
	ce(sr + 3) = mul * dot_2_constraintEquation(v3, v3g, spi, gi);
	ce(sr + 4) = mul * dot_1_constraintEquation(v3f, j_ptr->toGlobal(fj)/* Aj * fj*/);
}

void xTranslationConstraint::ConstraintJacobian(xSparseD& lhs, xVectorD& q, xVectorD& qd, unsigned int sr)
{
	unsigned int sc = 0;
	//unsigned int idx = i * xModel::OneDOF();
	//unsigned int jdx = j * xModel::OneDOF();
	//vector3d ri = new_vector3d(q(idx + 0), q(idx + 1), q(idx + 2));
	//vector3d rj = new_vector3d(q(jdx + 0), q(jdx + 1), q(jdx + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(jdx + 3), q(jdx + 4), q(jdx + 5), q(jdx + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
//	vector3d s;
	vector3d dij 
		= (j_ptr->Position() + Aj * spj) 
		- (i_ptr->Position() + Ai * spi);
	if (i)
	{
		sc = (i - 1) * xModel::OneDOF();
		lhs.insert(sr + 0, sc + 3, dot_1_constraintJacobian(Aj * hj, fi, ei));
		lhs.insert(sr + 1, sc + 3, dot_1_constraintJacobian(Aj * hj, gi, ei));
		lhs.insert(sr + 2, sc + 0, -fi, dot_2_constraintJacobian(dij + Ai * spi, fi, ei));
		lhs.insert(sr + 3, sc + 0, -gi, dot_2_constraintJacobian(dij + Ai * spi, gi, ei));
		lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(Aj * fj, fi, ei));	
	}
	if (j)
	{
		sc = (j - 1) * xModel::OneDOF();
		lhs.insert(sr + 0, sc + 3, dot_1_constraintJacobian(Ai * fi, hj, ej));
		lhs.insert(sr + 1, sc + 3, dot_1_constraintJacobian(Ai * gi, hj, ej));
		lhs.insert(sr + 2, sc + 0, Ai * fi, dot_2_constraintJacobian(Ai * fi, spj, ej));
		lhs.insert(sr + 3, sc + 0, Ai * gi, dot_2_constraintJacobian(Ai * gi, spj, ej));
		lhs.insert(sr + 4, sc + 3, dot_1_constraintJacobian(Ai * fi, fj, ej));
	}
}

void xTranslationConstraint::DerivateJacobian(xMatrixD& lhs, xVectorD& q, xVectorD& qd, double* lm, unsigned int sr, double mul)
{
	//unsigned int si = i * xModel::OneDOF();
	//unsigned int sj = j * xModel::OneDOF();
	//vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	//vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	vector3d dij = (j_ptr->Position() + Aj * spj) - (i_ptr->Position() + Ai * spi);
	dot_1_differentialJacobian(lhs, fi, hj, ei, ej, mul * lm[0]);
	dot_1_differentialJacobian(lhs, gi, hj, ei, ej, mul * lm[1]);
	dot_2_differentialJacobian(lhs, fi, spi, spj, dij, ei, ej, mul * lm[2]);
	dot_2_differentialJacobian(lhs, gi, spi, spj, dij, ei, ej, mul * lm[3]);
	dot_1_differentialJacobian(lhs, fi, fj, ei, ej, mul * lm[4]);
}

void xTranslationConstraint::SaveStepResult(
	unsigned int part, double ct, xVectorD& q, xVectorD& qd, double* L, unsigned int sr)
{
	unsigned int sc = 0;
// 	unsigned int idx = i * xModel::OneDOF();
// 	unsigned int jdx = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// new_vector3d(q(idx + 0), q(idx + 1), q(idx + 2));
	vector3d rj = j_ptr->Position();// new_vector3d(q(jdx + 0), q(jdx + 1), q(jdx + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(idx + 3), q(idx + 4), q(idx + 5), q(idx + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(jdx + 3), q(jdx + 4), q(jdx + 5), q(jdx + 6));
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
// 	vector3d s;
// 	vector4d r;
	vector3d dij = (rj + Aj * spj) - (ri + Ai * spi);
	kinematicConstraint_result kcr = { 0, };
	vector4d rf;
	vector3d f0, f1;
	vector4d r0, r1, r2, r3, r4;
	double *lm = L + sr;
	if (i)
	{
		r0 = -lm[0] * dot_1_constraintJacobian(Aj * hj, fi, ei);
		r1 = -lm[1] * dot_1_constraintJacobian(Aj * fj, gi, ei);
		f0 = lm[2] * fi;
		r2 = -lm[2] * dot_2_constraintJacobian(dij + Ai * spi, fi, ei);
		f1 = lm[3] * gi;
		r3 = -lm[3] * dot_2_constraintJacobian(dij + Ai * spi, gi, ei);
		r4 = -lm[4] * dot_1_constraintJacobian(Aj * fj, fi, ei);
		kcr.iaforce = f0 + f1;
		rf = r0 + r1 + r2 + r3 + r4;
		kcr.irforce = 0.5 * LMatrix(ei) * rf;
	}
	if (j)
	{
		r0 = -lm[0] * dot_1_constraintJacobian(Ai * fi, hj, ej);
		r1 = -lm[1] * dot_1_constraintJacobian(Ai * gi, hj, ej);
		f0 = lm[2] * fi;
		r2 = -lm[2] * dot_2_constraintJacobian(Ai * fi, spj, ej);
		f1 = lm[3] * gi;
		r3 = -lm[3] * dot_2_constraintJacobian(Ai * gi, spj, ej);
		r4 = -lm[4] * dot_1_constraintJacobian(Ai * fi, fj, ej);
		kcr.jaforce = f0 + f1;
		rf = r0 + r1 + r2 + r3 + r4;
		kcr.jrforce = 0.5 * LMatrix(ei) * rf;
	}
	kcr.time = ct;
	kcr.location = ri + Ai * spi;
	kcrs.push_back(kcr);
	nr_part++;
}

void xTranslationConstraint::GammaFunction(xVectorD& rhs, xVectorD& q, xVectorD& qd, unsigned int sr, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = i_ptr->Position();// new_vector3d(q(idx + 0), q(idx + 1), q(idx + 2));
	vector3d rj = j_ptr->Position();// new_vector3d(q(jdx + 0), q(jdx + 1), q(jdx + 2));
	euler_parameters ei = i_ptr->EulerParameters();// new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = j_ptr->EulerParameters();// q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters dei = i_ptr->DEulerParameters();// new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters dej = j_ptr->DEulerParameters();// ne
	vector3d dri = i_ptr->Velocity();
	vector3d drj = j_ptr->Velocity();
	matrix33d Ai = i_ptr->TransformationMatrix();// GlobalTransformationMatrix(ei);
	matrix33d Aj = j_ptr->TransformationMatrix();// GlobalTransformationMatrix(ej);
	vector3d dij = (rj + Aj * spj) - (ri + Ai * spi);
	rhs(sr + 0) = mul * dot_1_gamma(ei, ej, fi, hj, dei, dej);
	rhs(sr + 1) = mul * dot_1_gamma(ei, ej, gi, hj, dei, dej);
	rhs(sr + 2) = mul * dot_2_gamma(ei, ej, fi, spi, spj, dij, drj, dri, dei, dej);
	rhs(sr + 3) = mul * dot_2_gamma(ei, ej, gi, spi, spj, dij, drj, dri, dei, dej);
	rhs(sr + 4) = mul * dot_1_gamma(ei, ej, fi, fj, dei, dej);
}
