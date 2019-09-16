#include "xdynamics_object/xContactPairList.h"
#include "xdynamics_algebra/xAlgebraMath.h"
//#include <QtCore/QList>

xContactPairList::xContactPairList()
{

}

xContactPairList::~xContactPairList()
{
	plane_pair.delete_all();// qDeleteAll(plane_pair);
	particle_pair.delete_all();// qDeleteAll(particle_pair);
	cylinder_pair.delete_all();// qDeleteAll(cylinder_pair);
	triangle_pair.delete_all();
	triangle_line_pair.delete_all();
	triangle_point_pair.delete_all();
}

void xContactPairList::insertPlaneContactPair(xPairData* pd)
{
	if (plane_pair[pd->id])
		plane_pair[pd->id]->count += 1;
	else
		plane_pair.insert(pd->id, pd);
}

void xContactPairList::insertCylinderContactPair(xPairData * pd)
{
	cylinder_pair.insert(pd->id, pd);
}

void xContactPairList::insertParticleContactPair(xPairData* pd)
{
	particle_pair.insert(pd->id, pd);
}

void xContactPairList::insertTriangleContactPair(xTrianglePairData* pd)
{
	triangle_pair.insert(pd->id, pd);
}

void xContactPairList::insertTriangleLineContactPair(xTrianglePairData * pd)
{
	triangle_line_pair.insert(pd->id, pd);
}

void xContactPairList::insertTrianglePointContactPair(xTrianglePairData * pd)
{
	triangle_point_pair.insert(pd->id, pd);
}

void xContactPairList::deletePlanePairData(unsigned int i)
{
	if (plane_pair.find(i) != plane_pair.end())
	{
		xPairData* pd = plane_pair.take(i);
		delete pd;
	}	
}

void xContactPairList::deleteCylinderPairData(unsigned int i)
{
	if (cylinder_pair.find(i) != cylinder_pair.end())
	{
		xPairData* pd = cylinder_pair.take(i);
		delete pd;
	}
}

void xContactPairList::deleteParticlePairData(unsigned int i)
{
	if (particle_pair.size())
		i = i;
	if (particle_pair.find(i) != particle_pair.end())
	{
		xPairData* pd = particle_pair.take(i);
		delete pd;
	}	
}

void xContactPairList::deleteTrianglePairData(unsigned int i)
{
	if (triangle_pair.find(i) != triangle_pair.end())
	{
		xTrianglePairData* pd = triangle_pair.take(i);
		delete pd;
	}		
}

void xContactPairList::deleteTriangleLinePairData(unsigned int i)
{
	if (triangle_line_pair.find(i) != triangle_line_pair.end())
	{
		xTrianglePairData* pd = triangle_line_pair.take(i);
		delete pd;
	}
}

void xContactPairList::deleteTrianglePointPairData(unsigned int i)
{
	if (triangle_point_pair.find(i) != triangle_point_pair.end())
	{
		xTrianglePairData* pd = triangle_point_pair.take(i);
		delete pd;
	}
}

bool xContactPairList::IsNewPlaneContactPair(unsigned int i)
{
	return plane_pair.find(i) == plane_pair.end();
}

bool xContactPairList::IsNewCylinderContactPair(unsigned int i)
{
	return cylinder_pair.find(i) == cylinder_pair.end();
}

bool xContactPairList::IsNewParticleContactPair(unsigned int i)
{
	return particle_pair.find(i) == particle_pair.end();
}

bool xContactPairList::IsNewTriangleContactPair(unsigned int i)
{
	return triangle_pair.find(i) == triangle_pair.end();
}

bool xContactPairList::IsNewTriangleLineContactPair(unsigned int i)
{
	return triangle_line_pair.find(i) == triangle_line_pair.end();
}

bool xContactPairList::IsNewTrianglePointContactPair(unsigned int i)
{
	return triangle_point_pair.find(i) == triangle_point_pair.end();
}

bool xContactPairList::TriangleContactCheck(double r, vector3d& pos, xTrianglePairData * d)
{
	vector3d rp = new_vector3d(d->cpx, d->cpy, d->cpz) - pos;
	double dist = length(rp);
	bool b = r - dist;
	return b;
}

xmap<unsigned int, xPairData*>& xContactPairList::PlanePair()
{
	return plane_pair;
}

xmap<unsigned int, xPairData*>& xContactPairList::CylinderPair()
{
	return cylinder_pair;
}

xPairData* xContactPairList::PlanePair(unsigned int i)
{
	xmap<unsigned int, xPairData*>::iterator it = plane_pair.find(i);
	if (it == plane_pair.end())
		return NULL;
	return it.value();// plane_pair[i];
}

xmap<unsigned int, xPairData*>& xContactPairList::ParticlePair()
{
	return particle_pair;
}

xPairData* xContactPairList::ParticlePair(unsigned int i)
{
	xmap<unsigned int, xPairData*>::iterator it = particle_pair.find(i);
	if (it == particle_pair.end())
		return NULL;
	return it.value();// particle_pair[i];
}

xPairData * xContactPairList::CylinderPair(unsigned int i)
{
	xmap<unsigned int, xPairData*>::iterator it = cylinder_pair.find(i);
	if (it == cylinder_pair.end())
		return NULL;
	return it.value();// cylinder_pair[i];
}

xmap<unsigned int, xTrianglePairData*>& xContactPairList::TrianglePair()
{
	return triangle_pair;
}

xmap<unsigned int, xTrianglePairData*>& xContactPairList::TriangleLinePair()
{
	return triangle_line_pair;
}

xmap<unsigned int, xTrianglePairData*>& xContactPairList::TrianglePointPair()
{
	return triangle_point_pair;
}

xTrianglePairData* xContactPairList::TrianglePair(unsigned int i)
{
	xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_pair.find(i);
	if (it == triangle_pair.end())
		return NULL;

	return it.value();// triangle_pair[i];
}

xTrianglePairData * xContactPairList::TriangleLinePair(unsigned int i)
{
	xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_line_pair.find(i);
	if (it == triangle_line_pair.end())
		return NULL;

	return it.value();// triangle_line_pair[i];
}

xTrianglePairData * xContactPairList::TrianglePointPair(unsigned int i)
{
	xmap<unsigned int, xTrianglePairData*>::iterator it = triangle_point_pair.find(i);
	if (it == triangle_point_pair.end())
		return NULL;

	return it.value();// triangle_point_pair[i];
}
