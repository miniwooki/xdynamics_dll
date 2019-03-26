#ifndef XPARTICLEMESHOBJECTCONTACT_H
#define XPARTICLEMESHOBJECTCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xMeshObject.h"

class XDYNAMICS_API xParticleMeshObjectContact : public xContact
{
public:
	xParticleMeshObjectContact();
	xParticleMeshObjectContact(std::string _name);
	virtual ~xParticleMeshObjectContact();

	virtual void cudaMemoryAlloc(unsigned int np);
	//void insertContactParameters(unsigned int id, double r, double rt, double fr);
	xMeshObject* MeshObject() { return dynamic_cast<xMeshObject*>(po); }
	//virtual void updateCollisionPair(unsigned int id, xContactPairList& xcpl, double r, vector3d pos, double rj = 0, vector3d posj = new_vector3d(0.0, 0.0, 0.0));
	virtual void cuda_collision(
		double *pos, double *vel,
		double *omega, double *mass,
		double *force, double *moment,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

private:
	double maxRadii;
	xObject* p;
	xObject* po;
};


// #include "collision.h"
// 
// class polygonObject;
// class particle_system;
// 
// class collision_particles_polygonObject : public collision
// {
// public:
// 	collision_particles_polygonObject();
// 	collision_particles_polygonObject(QString& _name, modeler* _md, particle_system *_ps, polygonObject * _poly, tContactModel _tcm);
// 	virtual ~collision_particles_polygonObject();
// 
// 	virtual bool collid(double dt);
// 	virtual bool cuCollid(
// 		double *dpos /* = NULL */, double *dvel /* = NULL  */,
// 		double *domega /* = NULL */, double *dmass /* = NULL  */,
// 		double *dforce /* = NULL  */, double *dmoment /* = NULL */, unsigned int np);
// 	virtual bool collid_with_particle(unsigned int i, double dt);	
// 
// private:
// 	VEC3D particle_polygon_contact_detection(host_polygon_info& hpi, VEC3D& p, double pr);
// 	particle_system *ps;
// 	polygonObject *po;
// };
// 
#endif