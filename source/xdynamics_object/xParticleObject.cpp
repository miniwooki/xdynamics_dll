#include "..\..\include\xdynamics_object\xParticleObject.h"
#include "..\..\include\xdynamics_object\xParticleObject.h"
#include "xdynamics_object/xParticleObject.h"

unsigned int xParticleObject::xpo_count = 0;

xParticleObject::xParticleObject()
	: xObject()
	, sid(0)
	, np(0)
	, mid(0)
	, min_radius(0)
	, max_radius(0)
	, pos(NULL)
	, relative_loc(NULL)
	, each(1)
	, mass(0)
	, inertia(0)
{

}

xParticleObject::xParticleObject(std::string _name)
	: xObject(_name, PARTICLES)
	, sid(0)
	, np(0)
	, mid(0)
	, min_radius(0)
	, max_radius(0)
	, pos(NULL)
	, relative_loc(NULL)
	, each(1)
	, mass(0)
	, inertia(0)
{
	xObject::id = xpo_count;
	xpo_count++;
}

xParticleObject::~xParticleObject()
{
	if (pos) delete[] pos; pos = NULL;
	if (cpos) delete[] cpos; cpos = NULL;
	if (ep) delete[] ep; ep = NULL;
	if (mass) delete[] mass; mass = NULL;
	if (inertia) delete[] inertia; inertia = NULL;
	//if (relative_loc) delete[] relative_loc; relative_loc = NULL;
	xpo_count--;
}

void xParticleObject::setStartIndex(unsigned int _sid)
{
	sid = _sid;
}

void xParticleObject::setClusterStartIndex(unsigned int _csid)
{
	csid = _csid;
}

void xParticleObject::setEachCount(unsigned int ec)
{
	each = ec;
}

vector4d* xParticleObject::AllocMemory(unsigned int _np)
{
	np = _np;
	/*if(!pos)*/
	pos = new vector4d[np];
	//if (!mass)
	mass = new double[np];
	//if (!inertia)
	inertia = new vector3d[np];
	memset(pos, 0, sizeof(vector4d) * np);
	memset(mass, 0, sizeof(double) * np);
	memset(inertia, 0, sizeof(vector3d) * np);
	return pos;
}

//vector3d* xParticleObject::AllocInertiaMemory(unsigned int _np)
//{
//	if (inertia)
//		delete[] inertia;
//	inertia = new vector3d[_np];
//	return inertia;
//}

vector4d* xParticleObject::AllocClusterMemory(unsigned int _np)
{
	cnp = _np;
	if (!cpos)
	{
		cpos = new vector4d[_np];
		ep = new vector4d[_np];
		memset(cpos, 0, sizeof(vector4d) * _np);
		memset(ep, 0, sizeof(vector4d) * _np);
	}		
	return cpos;
}

void xParticleObject::CopyPosition(double* _pos)
{
	memcpy(_pos + sid * 4, pos, sizeof(vector4d) * np);
}

void xParticleObject::resizeParticles(unsigned int new_np)
{
	if (pos)
	{
		vector4d* tmp = new vector4d[np];
		memcpy(tmp, pos, sizeof(vector4d) * np);
		delete[] pos;
		pos = new vector4d[new_np];
		memcpy(pos, tmp, sizeof(vector4d) * np);
		delete[] tmp;
	}
	if (ep)
	{
		vector4d* tmp = new vector4d[np];
		memcpy(tmp, ep, sizeof(vector4d) * np);
		delete[] ep;
		ep = new vector4d[new_np];
		memcpy(ep, tmp, sizeof(vector4d) * np);
		delete[] tmp;
	}
	if (mass)
	{
		double* tmp = new double[np];
		memcpy(tmp, mass, sizeof(double) * np);
		delete[] mass;
		mass = new double[new_np];
		memcpy(mass, tmp, sizeof(double) * np);
		delete[] tmp;
	}
	if (inertia)
	{
		vector3d* tmp = new vector3d[np];
		memcpy(tmp, inertia, sizeof(vector3d) * np);
		delete[] inertia;
		inertia = new vector3d[new_np];
		memcpy(inertia, tmp, sizeof(vector3d) * np);
		delete[] tmp;
	}
	np = new_np;
}

void xParticleObject::CopyMassAndInertia(double * _mass, vector3d* _inertia)
{
	unsigned int _np = form == CLUSTER_SHAPE ? cnp : np;
	unsigned int _sid = form == CLUSTER_SHAPE ? csid : sid;
	memcpy(_mass + _sid, mass, sizeof(double) * _np);
	memcpy(_inertia + _sid, inertia, sizeof(vector3d) * _np);
	/*if (shape != NO_SHAPE_AND_MASS)
	{
		for (unsigned int i = 0; i < _np; i++)
		{
			_inertia[i] = inertia[i].x;
		}
	}
	else
	{
		memcpy(_inertia + _sid, inertia, sizeof(double) * _np * 3);
	}*/
}

void xParticleObject::CopyClusterPosition(unsigned int _sid, double* _pos, double *_ep)
{
	memcpy(_pos + _sid * 4, cpos, sizeof(vector4d) * cnp);
	memcpy(_ep + _sid * 4, ep, sizeof(vector4d) * cnp);
}

void xParticleObject::CopyClusterPosition(double* _pos, double *_ep)
{
	memcpy(_pos + csid * 4, cpos, sizeof(vector4d) * cnp);
	memcpy(_ep + csid * 4, ep, sizeof(vector4d) * cnp);
	/*for (unsigned int i = 0; i < cnp; i++)
	{
		for (unsigned int j = 0; j < each; j++)
		{
			cindex[(csid + i) * each + j] = csid + i;
		}
	}*/
}

unsigned int xParticleObject::StartIndex() const
{
	return sid;
}

unsigned int xParticleObject::StartClusterIndex() const
{
	return csid;
}

unsigned int xParticleObject::MassIndex() const
{
	return mid;
}

unsigned int xParticleObject::NumParticle() const
{
	return np;
}

unsigned int xParticleObject::NumCluster() const
{
	return cnp;
}

unsigned int xParticleObject::EachCount() const
{
	return each;
}

double xParticleObject::MinRadius() const
{
	return min_radius;
}

double xParticleObject::MaxRadius() const
{
	return max_radius;
}

xShapeType xParticleObject::ShapeForm() const
{
	return form;
}

vector4d* xParticleObject::Position() const
{
	return pos;
}

vector4d* xParticleObject::ClusterPosition() const
{
	return cpos;
}

vector4d * xParticleObject::EulerParameters() const
{
	return ep;
}

vector4d * xParticleObject::RelativeLocation() const
{
	return relative_loc;
}

double* xParticleObject::Mass() const
{
	return mass;
}

vector3d* xParticleObject::Inertia() const
{
	return inertia;
}

void xParticleObject::setParticleShapeName(xstring s)
{
	particleShapeName = s;
}

std::string xParticleObject::ParticleShapeName()
{
	return particleShapeName.toStdString();
}

unsigned int xParticleObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	return 0;
}

void xParticleObject::setMinRadius(double _mr)
{
	min_radius = _mr;
}

void xParticleObject::setMaxRadius(double _mr)
{
	max_radius = _mr;
}

void xParticleObject::setShapeForm(xShapeType xst)
{
	form = xst;
}

void xParticleObject::setRelativeLocation(vector4d * rl)
{
	relative_loc = rl;
}

void xParticleObject::setMassIndex(unsigned _mid)
{
	mid = _mid;
}


