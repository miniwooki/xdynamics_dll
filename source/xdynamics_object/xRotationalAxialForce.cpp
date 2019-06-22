#include "xdynamics_object/xRotationalAxialForce.h"

xRotationalAxialForce::xRotationalAxialForce()
	: xForce()
	, location(new_vector3d(0.0, 0.0, 0.0))
	, direction(new_vector3d(0.0, 0.0, 0.0))
	, r_force(0)
{
	memset(&f0_i.x, 0, sizeof(double) * 12);
}

xRotationalAxialForce::xRotationalAxialForce(std::string _name)
	: xForce(_name, RAXIAL)
	, location(new_vector3d(0.0, 0.0, 0.0))
	, direction(new_vector3d(0.0, 0.0, 0.0))
	, r_force(0)
{
	memset(&f0_i.x, 0, sizeof(double) * 12);
}

xRotationalAxialForce::~xRotationalAxialForce()
{

}

void xRotationalAxialForce::SetupDataFromStructure(xPointMass* ip, xPointMass* jp, xRotationalAxialForceData& d)
{
	xForce::i_ptr = ip;
	xForce::j_ptr = jp;
	location = new_vector3d(d.lx, d.ly, d.lz);
	direction = new_vector3d(d.dx, d.dy, d.dz);
	r_force = d.rforce;
	xForce::spi = i_ptr->toLocal(location - i_ptr->Position());
	xForce::spj = j_ptr->toLocal(location - j_ptr->Position());
}

void xRotationalAxialForce::xCalculateForce(const xVectorD& q, const xVectorD& qd)
{
	vector3d u;	
	if (i)
	{
		f0_i = f1_i;
		unsigned int sr = i * xModel::OneDOF();
		vector3d dp = i_ptr->Position() - location;
		u = cross(direction, dp);
		u = u / length(u);
		f1_i = (r_force / length(dp)) * u;
		vector3d mforce = cross(-dp, f1_i) + r_force * direction;
		i_ptr->addAxialForce(f1_i.x, f1_i.y, f1_i.z);
		i_ptr->addAxialMoment(mforce.x, mforce.y, mforce.z);
	}
	if (j)
	{
		f0_j = f1_j;
		unsigned int sr = j * xModel::OneDOF();
		vector3d dp = j_ptr->Position() - location;
		u = cross(direction, dp);
		u = u / length(u);
		f1_j = (r_force / length(dp)) * u;
		vector3d mforce = cross(-dp, f1_j) + r_force * direction;
		j_ptr->addAxialForce(f1_j.x, f1_j.y, f1_j.z);
		j_ptr->addAxialMoment(mforce.x, mforce.y, mforce.z);
	}
}

void xRotationalAxialForce::xDerivate(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{	
	double div9 = mul * r_force / 9.0;
	matrix33d m;
	m.a00 = 0;					m.a01 = -direction.z * div9; m.a02 = direction.y * div9;
	m.a10 = direction.z * div9;	m.a11 = 0;					 m.a12 = direction.x * div9;
	m.a20 = direction.y * div9;	m.a21 = direction.x * div9;	 m.a22 = 0;
	
	if (i)
	{
		unsigned int sr = (i - 1) * xModel::OneDOF();
		lhs.plus(sr, sr, m);
	}
	if (j)
	{
		unsigned int sr = (j - 1) * xModel::OneDOF();
		lhs.plus(sr, sr, m);
	}
}

void xRotationalAxialForce::xDerivateVelocity(xMatrixD& lhs, const xVectorD& q, const xVectorD& qd, double mul)
{

}