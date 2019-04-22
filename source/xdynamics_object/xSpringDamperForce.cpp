#include "xdynamics_object/xSpringDamperForce.h"

xSpringDamperForce::xSpringDamperForce()
	: xForce()
	, init_l(0.0)
	, k(0.0)
	, c(0.0)
	, L(new_vector3d(0.0, 0.0, 0.0))
	, f(0.0)
	, l(0.0)
	, dl(0.0)
{

}

xSpringDamperForce::xSpringDamperForce(std::string _name)
	: xForce(_name, TSDA)
	, init_l(0.0)
	, k(0.0)
	, c(0.0)
	, L(new_vector3d(0.0, 0.0, 0.0))
	, f(0.0)
	, l(0.0)
	, dl(0.0)
{

}

xSpringDamperForce::~xSpringDamperForce()
{

}

void xSpringDamperForce::SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xTSDAData& d)
{
	xForce::i_ptr = ip;
	xForce::j_ptr = jp;
	loc_i = new_vector3d(d.spix, d.spiy, d.spiz);
	loc_j = new_vector3d(d.spjx, d.spjy, d.spjz);
	k = d.k;
	c = d.c;
	init_l = d.init_l;
	xForce::spi = i_ptr->toLocal(loc_i - i_ptr->Position());
	xForce::spj = j_ptr->toLocal(loc_j - j_ptr->Position());
}

void xSpringDamperForce::xCalculateForce(const xVectorD& q, const xVectorD& qd)
{
	vector3d Qi;
	vector4d QRi;
	vector3d Qj;
	vector4d QRj;
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	vector3d vi = new_vector3d(qd(si + 0), qd(si + 1), qd(si + 2));
	vector3d vj = new_vector3d(qd(sj + 0), qd(sj + 1), qd(sj + 2));
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	L = rj + Aj * spj - ri - Ai * spi;
	//L = action->getPosition() + action->toGlobal(spj) - base->getPosition() - base->toGlobal(spi);
	l = length(L);// .length();
	vector3d dL = vj + BMatrix(edj, spj) * ej - vi - BMatrix(edi, spi) * ei;
	//VEC3D dL = action->getVelocity() + B(action->getEV(), spj) * action->getEP() - base->getVelocity() - B(base->getEV(), spi) * base->getEP();
	dl = dot(L, dL) / l;
	f = k * (l - init_l) + c * dl;
	Qi = (f / l) * L;
	Qj = -Qi;
	//matrix34d eee = BMatrix(ej, spj);
	QRi = (f / l) * BMatrix(ei, spi) * L;// transpose(B(base->getEP(), spi), L);
	QRj = -(f / l) * BMatrix(ej, spj) * L;// transpose(B(action->getEP(), spj), L);

	if (i)
	{
		//int irc = (i - 1) * xModel::OneDOF();
		i_ptr->addAxialForce(Qi.x, Qi.y, Qi.z);
		i_ptr->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
		//rhs.plus(irc, Qi, QRi);
	}
	if (j)
	{
		//int jrc = (j - 1) * xModel::OneDOF();// action->ID() * 7;
		j_ptr->addAxialForce(Qj.x, Qj.y, Qj.z);
		j_ptr->addEulerParameterMoment(QRj.x, QRj.y, QRj.z, QRj.w);
		//rhs.plus(jrc, Qj, QRj);
	}
}

void xSpringDamperForce::xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{
	matrix33d N1;
	matrix34d NP1;
	matrix33d N2;
	matrix34d NP2;
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	vector3d vi = new_vector3d(qd(si + 0), qd(si + 1), qd(si + 2));
	vector3d vj = new_vector3d(qd(sj + 0), qd(sj + 1), qd(sj + 2));
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix33d dAi = DGlobalTransformationMatrix(ei, edi);
	matrix33d dAj = DGlobalTransformationMatrix(ej, edj);
	matrix34d Bi = BMatrix(ei, spi);
	matrix34d Bj = BMatrix(ej, spj);
	double c1 = (k - (f / l) - c*(dl / l));
	double pow2l = 1.0 / pow(l, 2.0);
	matrix33d v1 = pow2l * L * L;
	vector3d v = vj + dAj * spj - vi - dAi * spi;// action->getVelocity() + action->toDGlobal(spj) - base->getVelocity() - base->toDGlobal(spi);
	matrix33d v2 = pow2l * L * v;// transpose(L, v) / pow(l, 2.0);
	matrix33d diag = { 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1 };
	N1 = c1 * v1 + c * v2 + (f / l) * diag;
	N2 = c * v1;
	NP1 = N1 * Bi + N2 * BMatrix(edi, spi);
	NP2 = N1 * Bj + N2 * BMatrix(edj, spj);
	//matrix34d N1BiN2Bd = N1 * Bi + N2 * BMatrix(edi, spi);
	//matrix34d N1BjN2Bd = N1 * Bj + N2 * BMatrix(edj, spj);
	int irc = (i - 1) * xModel::OneDOF();
	int jrc = (j - 1) * xModel::OneDOF();
	if (i)
	{
		lhs.plus(irc, irc, -mul * N1);
		lhs.plus(irc, irc + 3, -mul * NP1);
		lhs.plus(irc + 3, irc, -mul * Bi * N1);
		lhs.plus(irc + 3, irc + 3, -mul * Bi * NP1 + (f / l) * DMatrix(spi, L));
	}
	if (j)
	{
		lhs.plus(jrc, jrc, -mul * N1);
		lhs.plus(jrc, jrc + 3, -mul * NP2);
		lhs.plus(jrc + 3, jrc, -mul * Bj * N1);
		lhs.plus(jrc + 3, jrc + 3, -mul * Bj * NP2/* - (f / l) * DMatrix(spj, L)*/);
	}
	if (i && j)
	{
		lhs.plus(irc, jrc, mul * N1);
		lhs.plus(irc, jrc + 3, mul * NP1);
		lhs.plus(jrc, irc, mul * N1);
		lhs.plus(jrc, irc + 3, mul * NP1);
		lhs.plus(irc + 3, jrc, mul * Bi * N1);
		lhs.plus(irc + 3, jrc + 3, mul * Bi * NP2);
		lhs.plus(jrc + 3, irc, mul * Bj * N1);
		lhs.plus(jrc + 3, irc + 3, mul * Bj * NP1 - (f / l) * DMatrix(spj, L));
	}
}

void xSpringDamperForce::xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{
	unsigned int si = i * xModel::OneDOF();
	unsigned int sj = j * xModel::OneDOF();
	vector3d ri = new_vector3d(q(si + 0), q(si + 1), q(si + 2));
	vector3d rj = new_vector3d(q(sj + 0), q(sj + 1), q(sj + 2));
	vector3d vi = new_vector3d(qd(si + 0), qd(si + 1), qd(si + 2));
	vector3d vj = new_vector3d(qd(sj + 0), qd(sj + 1), qd(sj + 2));
	euler_parameters ei = new_euler_parameters(q(si + 3), q(si + 4), q(si + 5), q(si + 6));
	euler_parameters ej = new_euler_parameters(q(sj + 3), q(sj + 4), q(sj + 5), q(sj + 6));
	euler_parameters edi = new_euler_parameters(qd(si + 3), qd(si + 4), qd(si + 5), qd(si + 6));
	euler_parameters edj = new_euler_parameters(qd(sj + 3), qd(sj + 4), qd(sj + 5), qd(sj + 6));
	matrix33d Ai = GlobalTransformationMatrix(ei);
	matrix33d Aj = GlobalTransformationMatrix(ej);
	matrix33d dAi = DGlobalTransformationMatrix(ei, edi);
	matrix33d dAj = DGlobalTransformationMatrix(ej, edj);
	matrix34d Bi = BMatrix(ei, spi);
	matrix34d Bj = BMatrix(ej, spj);
	double pow2l = 1.0 / pow(l, 2.0);
	matrix33d coe = pow2l * c * L * L;// transpose(L, L) / pow(l, 2.0);
	matrix43d coeiN = pow2l * c * Bi * (L * L);// transpose(B(action->getEP(), spj), transpose(L, L) / pow(l, 2.0));
	matrix43d coejN = pow2l * c * Bj * (L * L);
	int irc = (i - 1) * xModel::OneDOF();
	int jrc = (j - 1) * xModel::OneDOF();
	if (i)
	{
		lhs.plus(irc, irc, -mul * coe);
		lhs.plus(irc, irc + 3, -mul * coe * Bi);
		lhs.plus(irc + 3, irc, -mul * coeiN);
		lhs.plus(irc + 3, irc + 3, -mul * coeiN * Bi);
	}

	if (j)
	{
		lhs.plus(jrc, jrc, -mul * coe);
		lhs.plus(jrc, jrc + 3, -mul * coe * Bj);
		lhs.plus(jrc + 3, jrc, -mul * coejN);
		lhs.plus(jrc + 3, jrc + 3, -mul * coejN * Bj);
	}
	if (i && j)
	{
		lhs.plus(irc, jrc, mul * coe);
		lhs.plus(irc, jrc + 3, mul * coe * Bj);
		lhs.plus(irc + 3, jrc, mul * coeiN);
		lhs.plus(irc + 3, jrc + 3, mul * coeiN * Bj);
		lhs.plus(jrc, irc, mul * coe);
		lhs.plus(jrc, irc + 3, mul * coe * Bi);
		lhs.plus(jrc + 3, irc, mul * coejN);
		lhs.plus(jrc + 3, irc + 3, mul * coejN * Bi);		
	}
}