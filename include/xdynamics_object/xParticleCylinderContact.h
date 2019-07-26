#ifndef XPARTICLECYLINDERCONTACT_H
#define XPARTICLECYLINDERCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xCylinderObject.h"
//#include "xdynamics_object/xParticlePlaneContact.h"
#include <QtCore/QMap>
#include <QtCore/QString>

//class xPlaneObject;
class xParticleObject;
//class xCylinderObject;

class XDYNAMICS_API xParticleCylinderContact : public xContact
{
	
	enum cc_contact_type{NO_CCT = 0, RADIAL_WALL_CONTACT, BOTTOM_OR_TOP_CIRCLE_CONTACT, CIRCLE_LINE_CONTACT };
public:
	xParticleCylinderContact();
	xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleCylinderContact();

	// 	double MaxRadiusOfPolySphere() { return maxRadius; }
	// 	double* SphereData() { return dsphere; }
	// 	vector4d* HostSphereData() { return hsphere; }
	/*void define(unsigned int i, xParticlePlaneContact* d);
	void define(unsigned int i, xParticleCubeContact* d);*/
	//void allocHostMemory(unsigned int n);
	/*bool pcylCollision(
		xContactPairList* pairs, unsigned int i, double r, double m,
		vector3d& p, vector3d& v, vector3d& o,
		double &R, vector3d& T, vector3d& F, vector3d& M,
		unsigned int nco, xClusterInformation* xci, vector4d* cpos);*/
	//unsigned int NumContact();// { return ncontact; }
//	void setNumContact(unsigned int c) { ncontact = c; }
	//void updateMeshObjectData(xVectorD& q, xVectorD& qd);
	//bool updateCollisionPair(xContactPairList& xcpl, double r, vector3d pos);
	xCylinderObject* CylinderObject();
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);
	//void setZeroCollisionForce();
	//device_cylinder_info* deviceCylinderInfo();
	//unsigned int NumPlanes();

private:

	//vector3d particle_polygon_contact_detection(host_mesh_info& dpi, vector3d& p, double r/*, polygonContactType& _pct*/);
	
	//unsigned int nplanes;
	// 	unsigned int *d_old_pair_count;
	// 	unsigned int *d_pair_count;
	// 	unsigned int *d_old_pair_start;
	// 	unsigned int *d_pair_start;
		/*particle_plane_pair_data* d_pppd;*/
	//xContactMaterialParameters* hcmp;
	//xMaterialPair* xmps;
	//'//host_cylinder_info* hpi;
	//bool isInerContact;
	xCylinderObject::empty_part empty_cylinder_part;
	xParticleObject* p_ptr;
	xCylinderObject* c_ptr;
	
	cc_contact_type cct;
};

#endif