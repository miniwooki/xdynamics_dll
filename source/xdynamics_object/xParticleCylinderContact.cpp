#include "xdynamics_object/xParticleCylinderContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xCylinderObject.h"
#include "xdynamics_simulation/xSimulation.h"

xParticleCylinderContact::xParticleCylinderContact()
	: xContact()
	, p_ptr(NULL)
	, c_ptr(NULL)
//	, dci(NULL)
{

}

xParticleCylinderContact::xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_CYLINDER)
	, p_ptr(NULL)
	, c_ptr(NULL)
//	, dci(NULL)
{
	if (o1->Shape() == CYLINDER_SHAPE)
	{
		c_ptr = dynamic_cast<xCylinderObject*>(o1);
		p_ptr = dynamic_cast<xParticleObject*>(o2);
	}
	else
	{
		c_ptr = dynamic_cast<xCylinderObject*>(o2);
		p_ptr = dynamic_cast<xParticleObject*>(o1);
	}
	//hci = { c_ptr->cylinder_length(), c_ptr->cylinder_bottom_radius(), c_ptr->cylinder_top_radius(), c_ptr->bottom_position(), c_ptr->top_position() };
	empty_cylinder_part = c_ptr->empty_part_type();
}

xParticleCylinderContact::~xParticleCylinderContact()
{

}

// void xParticleCylinderContact::allocHostMemory(unsigned int n)
// {
// 
// }





xCylinderObject * xParticleCylinderContact::CylinderObject()
{
	return c_ptr;
}

void xParticleCylinderContact::cudaMemoryAlloc(unsigned int np)
{
	xContact::cudaMemoryAlloc(np);
	if (xSimulation::Gpu())
	{

	}
}

void xParticleCylinderContact::cuda_collision(double *pos, double *vel, double *omega, double *mass, double *force, double *moment, unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np)
{

}

//device_cylinder_info* xParticleCylinderContact::deviceCylinderInfo()
//{
//	return dci;
//}


