#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include <QtCore/QVector>

xSmoothedParticleHydrodynamicsModel *xsph_ptr = NULL;

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel()
	: xpmgr(NULL)
	, bound(DUMMY_PARTICLE_METHOD)
{
	xpmgr = new xParticleManager;
	xsph_ptr = this;
}

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel(std::string _name)
	: name(QString::fromStdString(_name))
	, xpmgr(NULL)
	, bound(DUMMY_PARTICLE_METHOD)
{
	xpmgr = new xParticleManager;
	xsph_ptr = this;
}

xSmoothedParticleHydrodynamicsModel::~xSmoothedParticleHydrodynamicsModel()
{
	if (xpmgr) delete xpmgr; xpmgr = NULL;
}

xSmoothedParticleHydrodynamicsModel* xSmoothedParticleHydrodynamicsModel::XSPH()
{
	return xsph_ptr;
}

xParticleManager* xSmoothedParticleHydrodynamicsModel::XParticleManager()
{
	return xpmgr;
}

void xSmoothedParticleHydrodynamicsModel::setParticleSpacing(double ps)
{
	pspace = ps;
}

void xSmoothedParticleHydrodynamicsModel::setFreeSurfaceFactor(double fs)
{
	fs_factor = fs;
}

void xSmoothedParticleHydrodynamicsModel::setReferenceDensity(double d)
{
	ref_rho = d;
}

void xSmoothedParticleHydrodynamicsModel::setKinematicViscosity(double v)
{
	k_viscosity = v;
}

void xSmoothedParticleHydrodynamicsModel::setKernelFunctionData(xKernelFunctionData& d)
{
	ker_data = d;
}

xBoundaryTreatmentType xSmoothedParticleHydrodynamicsModel::BoundaryTreatmentType()
{
	return bound;
}

bool xSmoothedParticleHydrodynamicsModel::CheckCorner(vector3d p)
{
	foreach(xOverlapCorner xoc, overlappingCorners)
	{
		vector3d position = new_vector3d(xoc.c1.px, xoc.c1.py, xoc.c1.pz);
		if (length(position - p) < 1e-9)
			return true;
	}
	return false;
}

void xSmoothedParticleHydrodynamicsModel::DefineCorners(xObjectManager* xobj)
{
	QVector<xCorner> corners;
	bool exist_corner3 = false;
	foreach(xObject* xo, xobj->XObjects())
	{
		if (xo->Material() != BOUNDARY)
			continue;
		QVector<xCorner> objCorners = xo->get_sph_boundary_corners();
		for (unsigned int i = 0; i < objCorners.size(); i++)
		{
			for (unsigned int j = 0; j < corners.size(); j++)
			{
				xCorner xc0 = objCorners[i];
				xCorner xc1 = corners[j];
				vector3d p0 = new_vector3d(xc0.px, xc0.py, xc0.pz);
				vector3d p1 = new_vector3d(xc1.px, xc1.py, xc1.pz);
				if (length(p0 - p1) < 1e-9) // if same position between two corners
				{
					if (xo->Shape() == PLANE_SHAPE){
						// if geometry is plane
						for (unsigned int k = 0; k < overlappingCorners.size(); k++){
							xOverlapCorner c = overlappingCorners[k];
							p1 = new_vector3d(c.c1.px, c.c1.py, c.c1.pz);
							if (length(p1 - p0) < 1e-9){
								p1 = new_vector3d(c.c2.px, c.c2.py, c.c2.pz);
								if (length(p1 - p0) < 1e-9){
									overlappingCorners[k].c3 = objCorners[i];
									overlappingCorners[k].cnt = 1;
									exist_corner3 = true;
								}
							}
						}
					}
					if (!exist_corner3){
						xOverlapCorner c = { 0, 0, 0, 0, 0, objCorners[i], corners[j], corners[i], false };
						overlappingCorners.push_back(c);
					}
					else{
						exist_corner3 = false;
					}
					break;
				}
			}
			corners.push_back(objCorners[i]);
		}
	}
}

void xSmoothedParticleHydrodynamicsModel::CreateParticles(xObjectManager* xobj)
{
	DefineCorners(xobj);
	foreach(xObject* xo, xobj->XObjects())
	{
		xpmgr->CreateSPHParticles(xo, pspace);
	}
}
