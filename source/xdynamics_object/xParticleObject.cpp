#include "xdynamics_object/xParticleObject.h"

unsigned int xParticleObject::xpo_count = 0;

xParticleObject::xParticleObject()
	: xObject()
	, sid(0)
	, np(0)
	, min_radius(0)
	, max_radius(0)
	, pos(NULL)
	, relative_loc(NULL)
	, each(1)
{

}

xParticleObject::xParticleObject(std::string _name)
	: xObject(_name, PARTICLES)
	, sid(0)
	, np(0)
	, min_radius(0)
	, max_radius(0)
	, pos(NULL)
	, relative_loc(NULL)
	, each(1)
{
	xObject::id = xpo_count;
	xpo_count++;
}

xParticleObject::~xParticleObject()
{
	if (pos) delete[] pos; pos = NULL;
	if (cpos) delete[] cpos; cpos = NULL;
	xpo_count--;
}

void xParticleObject::setStartIndex(unsigned int _sid)
{
	sid = _sid;
}

void xParticleObject::setClusterStartIndex(unsigned int _csid)
{
	csid = csid;
}

void xParticleObject::setEachCount(unsigned int ec)
{
	each = ec;
}

vector4d* xParticleObject::AllocMemory(unsigned int _np)
{
	np = _np;
	if(!pos)
		pos = new vector4d[np];
	return pos;
}

vector4d* xParticleObject::AllocClusterMemory(unsigned int _np)
{
	cnp = _np;
	if (!cpos)
	{
		cpos = new vector4d[_np];
		ep = new vector4d[_np];
	}
		
	return cpos;
}

void xParticleObject::CopyPosition(double* _pos)
{
	memcpy(_pos + sid * 4, pos, sizeof(vector4d) * np);
}

void xParticleObject::CopyClusterPosition(double* _pos, double *_ep, unsigned int* cindex)
{
	memcpy(_pos + csid * 4, cpos, sizeof(vector4d) * cnp);
	memcpy(_ep + csid * 4, ep, sizeof(vector4d) * cnp);
	for (unsigned int i = 0; i < cnp; i++)
	{
		for (unsigned int j = 0; j < each; j++)
		{
			cindex[(csid + i) * each + j] = csid + i;
		}
	}
}

unsigned int xParticleObject::StartIndex() const
{
	return sid;
}

unsigned int xParticleObject::NumParticle() const
{
	return np;
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

vector4d * xParticleObject::EulerParameters() const
{
	return ep;
}

vector3d * xParticleObject::RelativeLocation() const
{
	return relative_loc;
}

unsigned int xParticleObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	return 0;
}

QVector<xCorner> xParticleObject::get_sph_boundary_corners()
{
	return QVector<xCorner>();
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

void xParticleObject::setRelativeLocation(vector3d * rl)
{
	relative_loc = rl;
}
