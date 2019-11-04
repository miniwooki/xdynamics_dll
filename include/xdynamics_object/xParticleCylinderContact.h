#ifndef XPARTICLECYLINDERCONTACT_H
#define XPARTICLECYLINDERCONTACT_H

#include "xdynamics_object/xConatct.h"
#include "xdynamics_object/xCylinderObject.h"
//#include "xdynamics_object/xParticlePlaneContact.h"
//#include <QtCore/QMap>
//#include <QtCore/QString>

//class xPlaneObject;
class xParticleObject;
//class xCylinderObject;

class XDYNAMICS_API xParticleCylinderContact : public xContact
{
	enum cc_contact_type { NO_CCT = 0, RADIAL_WALL_CONTACT, BOTTOM_OR_TOP_CIRCLE_CONTACT, CIRCLE_LINE_CONTACT };
	struct host_cylinder_info
	{
		unsigned int id;
		unsigned int empty_part;
		double thickness;
		vector3d len_rr;
		vector3d pbase;
		vector3d ptop;
	};	
public:
	xParticleCylinderContact();
	xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleCylinderContact();
	
	xCylinderObject* CylinderObject();
	virtual void collision(
		double *pos, double *ep, double *vel, double *ev,
		double *mass, double* inertia,
		double *force, double *moment,
		double *tmax, double* rres,
		unsigned int *sorted_id,
		unsigned int *cell_start,
		unsigned int *cell_end,
		unsigned int np);

	virtual void define(unsigned int idx, unsigned int np);
	virtual void update();

	static void savePartData(unsigned int np);

private:
	bool allocated_static;
	unsigned int id;
	static unsigned int *d_pair_count_pcyl;
	static unsigned int *d_pair_id_pcyl;
	static double *d_tsd_pcyl;

	static unsigned int* pair_count_pcyl;
	static unsigned int* pair_id_pcyl;
	static double* tsd_pcyl;
	host_cylinder_info hci;

	device_cylinder_info* dci;
	device_body_info* dbi;
	//device_body_force* dbf;

	xCylinderObject::empty_part empty_cylinder_part;
	xParticleObject* p_ptr;
	xCylinderObject* c_ptr;
	cc_contact_type cct;
};

#endif