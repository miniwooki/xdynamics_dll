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
	
	enum cc_contact_type{NO_CCT = 0, RADIAL_WALL_CONTACT, BOTTOM_OR_TOP_CIRCLE_CONTACT, CIRCLE_LINE_CONTACT };
public:
	xParticleCylinderContact();
	xParticleCylinderContact(std::string _name, xObject* o1, xObject* o2);
	virtual ~xParticleCylinderContact();

	xCylinderObject* CylinderObject();
	virtual void cudaMemoryAlloc(unsigned int np);
	virtual void cuda_collision(
		double *pos, double *vel, double *omega,
		double *mass, double *force, double *moment,
		unsigned int *sorted_id, unsigned int *cell_start, unsigned int *cell_end, unsigned int np);

private:
	xCylinderObject::empty_part empty_cylinder_part;
	xParticleObject* p_ptr;
	xCylinderObject* c_ptr;
	
	cc_contact_type cct;
};

#endif