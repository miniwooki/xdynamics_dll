#include "xdynamics_object/xParticleCylinderContact.h"
#include "xdynamics_object/xParticleObject.h"
#include "xdynamics_object/xCylinderObject.h"
#include "xdynamics_simulation/xSimulation.h"
#include "xdynamics_manager/xDynamicsManager.h"

double* xParticleCylinderContact::d_tsd_pcyl = nullptr;
unsigned int* xParticleCylinderContact::d_pair_count_pcyl = nullptr;
unsigned int* xParticleCylinderContact::d_pair_id_pcyl = nullptr;

double* xParticleCylinderContact::tsd_pcyl = nullptr;
unsigned int* xParticleCylinderContact::pair_count_pcyl = nullptr;
unsigned int* xParticleCylinderContact::pair_id_pcyl = nullptr;

xParticleCylinderContact::xParticleCylinderContact()
	: xContact()
	, id(0)
	, p_ptr(NULL)
	, c_ptr(NULL)
	, allocated_static(false)
{

}

xParticleCylinderContact::xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2)
	: xContact(_name, PARTICLE_CYLINDER)
	, id(0)
	, p_ptr(NULL)
	, c_ptr(NULL)
	, allocated_static(false)
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
	mpp = { o1->Youngs(), o2->Youngs(), o1->Poisson(), o2->Poisson(), o1->Shear(), o2->Shear() };
	empty_cylinder_part = c_ptr->empty_part_type();
}

xParticleCylinderContact::~xParticleCylinderContact()
{
	if (d_pair_count_pcyl) checkCudaErrors(cudaFree(d_pair_count_pcyl)); d_pair_count_pcyl = NULL;
	if (d_pair_id_pcyl) checkCudaErrors(cudaFree(d_pair_id_pcyl)); d_pair_id_pcyl = NULL;
	if (d_tsd_pcyl) checkCudaErrors(cudaFree(d_tsd_pcyl)); d_tsd_pcyl = NULL;
	if (dci) checkXerror(cudaFree(dci)); dci = NULL;
	if (dbi) checkXerror(cudaFree(dbi)); dbi = NULL;
	//if (dbf) checkXerror(cudaFree(dci)); dci = NULL;

	if (pair_count_pcyl) delete[] pair_count_pcyl; pair_count_pcyl = NULL;
	if (pair_id_pcyl) delete[] pair_id_pcyl; pair_id_pcyl = NULL;
	if (tsd_pcyl) delete[] tsd_pcyl; tsd_pcyl = NULL;
}

void xParticleCylinderContact::define(unsigned int idx, unsigned int np)
{
	id = idx;
	xContact::define(idx, np);
	hci =
	{
		idx,
		(unsigned int)c_ptr->empty_part_type(),
		c_ptr->cylinder_thickness(),
		c_ptr->cylinder_length(),
		c_ptr->cylinder_bottom_radius(),
		c_ptr->cylinder_top_radius(),
		c_ptr->bottom_position(),
		c_ptr->top_position()
	};
	if (xSimulation::Gpu())
	{		
		if (!allocated_static)
		{
			pair_count_pcyl = new unsigned int[np];
			pair_id_pcyl = new unsigned int[np * MAX_P2CY_COUNT];
			tsd_pcyl = new double[2 * np * MAX_P2CY_COUNT];
			checkXerror(cudaMalloc((void**)&d_pair_count_pcyl, sizeof(unsigned int) * np));
			checkXerror(cudaMalloc((void**)&d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMalloc((void**)&d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMemset(d_pair_count_pcyl, 0, sizeof(unsigned int) * np));
			checkXerror(cudaMemset(d_pair_id_pcyl, 0, sizeof(unsigned int) * np * MAX_P2CY_COUNT));
			checkXerror(cudaMemset(d_tsd_pcyl, 0, sizeof(double2) * np * MAX_P2CY_COUNT));
		}
		
		checkXerror(cudaMalloc((void**)&dci, sizeof(device_cylinder_info)));
		checkXerror(cudaMalloc((void**)&dbi, sizeof(device_body_info)));
		checkXerror(cudaMemset(dbi, 0, sizeof(device_body_info)));
		checkXerror(cudaMemcpy(dci, &hci, sizeof(device_cylinder_info), cudaMemcpyHostToDevice));
		xDynamicsManager::This()->XResult()->set_p2cyl_contact_data((int)MAX_P2CY_COUNT);
		update();
	}
}

void xParticleCylinderContact::update()
{
	//device_body_info *bi = NULL;
	//if (nContactObject || is_first_set_up)
	//	bi = new device_body_info[nContactObject];


	if (xSimulation::Gpu())
	{
		//unsigned int mcnt = 0;
		//for (xmap<unsigned int, xContact*>::iterator it = pair_contact.begin(); it != pair_contact.end(); it.next())
		//{
			//xPointMass* pm = NULL;
			//xContact* xc = it.value();
			//xCylinderObject *c = dynamic_cast<xParticleCylinderContact*>(xc)->CylinderObject();
		euler_parameters ep = c_ptr->EulerParameters();
		euler_parameters ed = c_ptr->DEulerParameters();
		host_body_info hbi = {
			c_ptr->Mass(),
			c_ptr->Position().x, c_ptr->Position().y, c_ptr->Position().z,
			c_ptr->Velocity().x, c_ptr->Velocity().y, c_ptr->Velocity().z,
			ep.e0, ep.e1, ep.e2, ep.e3,
			ed.e0, ed.e1, ed.e2, ed.e3
		};
		
		//checkCudaErrors(cudaMemset(db_force, 0, sizeof(double3) * ncylinders));
		//checkCudaErrors(cudaMemset(db_moment, 0, sizeof(double3) * ncylinders));
		checkXerror(cudaMemcpy(dbi, &hbi, sizeof(device_body_info), cudaMemcpyHostToDevice));
	}
}

void xParticleCylinderContact::savePartData(unsigned int np)
{
	checkXerror(cudaMemcpy(pair_count_pcyl, d_pair_count_pcyl, sizeof(unsigned int) * np, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(pair_id_pcyl, d_pair_id_pcyl, sizeof(unsigned int) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
	checkXerror(cudaMemcpy(tsd_pcyl, d_tsd_pcyl, sizeof(double2) * np * MAX_P2CY_COUNT, cudaMemcpyDeviceToHost));
	xDynamicsManager::This()->XResult()->save_p2cyl_contact_data(pair_count_pcyl, pair_id_pcyl, tsd_pcyl);
}


xCylinderObject * xParticleCylinderContact::CylinderObject()
{
	return c_ptr;
}



void xParticleCylinderContact::collision(
	double *pos, double *ep, double *vel, double *ev,
	double *mass, double* inertia,
	double *force, double *moment,
	double *tmax, double* rres,
	unsigned int *sorted_id,
	unsigned int *cell_start,
	unsigned int *cell_end,
	unsigned int np)
{
	if (xSimulation::Gpu())
	{
		double fm[6] = { 0, };
		cu_cylinder_contact_force(
			dci, dbi, dcp,
			pos, ep, vel, ev, force, moment, mass,
			tmax, rres,
			d_pair_count_pcyl, d_pair_id_pcyl, d_tsd_pcyl, np);
		if (c_ptr->isDynamicsBody())
		{
			fm[0] = reduction(xContact::deviceBodyForceX(), np);
			fm[1] = reduction(xContact::deviceBodyForceY(), np);
			fm[2] = reduction(xContact::deviceBodyForceZ(), np);
			fm[3] = reduction(xContact::deviceBodyMomentX(), np);
			fm[4] = reduction(xContact::deviceBodyMomentY(), np);
			fm[5] = reduction(xContact::deviceBodyMomentZ(), np);
			c_ptr->addAxialForce(fm[0], fm[1], fm[2]);
			c_ptr->addAxialMoment(fm[3], fm[4], fm[5]);
		}
	}
	
}

//device_cylinder_info* xParticleCylinderContact::deviceCylinderInfo()
//{
//	return dci;
//}


