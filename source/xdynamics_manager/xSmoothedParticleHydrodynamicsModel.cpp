#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include <QtCore/QVector>

xSmoothedParticleHydrodynamicsModel *xsph_ptr = NULL;

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel()
	: name("")
	, bound(DUMMY_PARTICLE_METHOD)
	, pos(NULL)
	, vel(NULL)
	, type(NULL)
	, nfluid(0)
	, nbound(0)
	, ndummy(0)
	, nlayers(3)
{
	xsph_ptr = this;
}

xSmoothedParticleHydrodynamicsModel::xSmoothedParticleHydrodynamicsModel(std::string _name)
	: name(QString::fromStdString(_name))
	, bound(DUMMY_PARTICLE_METHOD)
	, pos(NULL)
	, vel(NULL)
	, type(NULL)
	, nfluid(0)
	, nbound(0)
	, ndummy(0)
	, nlayers(3)
{
	xsph_ptr = this;
}

xSmoothedParticleHydrodynamicsModel::~xSmoothedParticleHydrodynamicsModel()
{
	//if (xpmgr) delete xpmgr; xpmgr = NULL;
	//if (all_particles) delete[] all_particles; all_particles = NULL;
	if (pos) delete[] pos; pos = NULL;
	if (vel) delete[] vel; vel = NULL;
	if (type) delete[] type; type = NULL;
}

xSmoothedParticleHydrodynamicsModel* xSmoothedParticleHydrodynamicsModel::XSPH()
{
	return xsph_ptr;
}

// xParticleManager* xSmoothedParticleHydrodynamicsModel::XParticleManager()
// {
// 	return xpmgr;
// }

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

xSPHCorrectionType xSmoothedParticleHydrodynamicsModel::CorrectionType()
{
	return corr;
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
		for (int i = 0; i < objCorners.size(); i++)
		{
			for (int j = 0; j < corners.size(); j++)
			{
				xCorner xc0 = objCorners[i];
				xCorner xc1 = corners[j];
				vector3d p0 = new_vector3d(xc0.px, xc0.py, xc0.pz);
				vector3d p1 = new_vector3d(xc1.px, xc1.py, xc1.pz);
				if (length(p0 - p1) < 1e-9) // if same position between two corners
				{
					if (xo->Shape() == PLANE_SHAPE){
						// if geometry is plane
						for (int k = 0; k < overlappingCorners.size(); k++){
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
	unsigned int count = 0;
	foreach(xObject* xo, xobj->XObjects())
		count += xo->create_sph_particles(pspace, nlayers);
	np = count;
	np += CreateOverlapCornerDummyParticles(np, false);
	if (!pos) pos = new vector3d[np];
	if (!vel) vel = new vector3d[np];
	if (!type) type = new xMaterialType[np];
	count = 0;
	foreach(xObject* xo, xobj->XObjects())
		count += xo->create_sph_particles(pspace, nlayers, pos, type);
	count += CreateOverlapCornerDummyParticles(count, true);
	for (unsigned int i = 0; i < np; i++)
	{
		switch (type[i])
		{
		case FLUID: nfluid++; break;
		case BOUNDARY: nbound++; break;
		case DUMMY: ndummy++; break;
		}
	}
	dim = ker_data.dim;
	p_volume = pow(pspace, dim);
	p_mass = p_volume * ref_rho;
// 	foreach(xParticleObject* xpo, xpmgr->XParticleObjects())
// 	{
// 		if (xpo->Material() == FLUID)
// 			nfluid += xpo->NumParticle();
// 		else if (xpo->Material() == BOUNDARY)
// 		{
// 			unsigned int nb = xpo->NumParticle();
// 			nbound += nb;
// 			ndummy += nb * nlayers;
// 		}
// 	}
// 	CreateOverlapCornerDummyParticles(false);
// 	np = nfluid + nbound + ndummy + noverlap;
// 	if (!all_particles)
// 		all_particles = new vector3d[np];
// 	nfluid = nbound = ndummy = noverlap = 0;
// 	foreach(xParticleObject* xpo, xpmgr->XParticleObjects())
// 	{
// 		if (xpo->Material() == FLUID)
// 		{
// 			vector4d* p = xpo->Position();
// 			nfluid = xpo->NumParticle();
// 			for (unsigned int i = 0; i < nfluid; i++)
// 			{
// 				all_particles[i] = new_vector3d(p[i].x, p[i].y, p[i].z);
// 			}
// 		}
// 		else if (xpo->Material() == BOUNDARY)
// 		{
// 			unsigned int nb = xpo->NumParticle();
// 			unsigned int sid = nfluid + nbound;
// 			vector3d n = xobj->XObjects()[xpo->Name()]->
// 			nbound += nb;
// 			vector4d* p = xpo->Position();
// 			for (unsigned int i = sid; i < sid + nb; i++)
// 			{
// 				all_particles[i + ndummy] = new_vector3d(p[i].x, p[i].y, p[i].z);
// 				for (unsigned int j = i; j < i + nlayers; i++)
// 				{
// 					all_particles[j] = new_vector3d
// 				}
// 			}
// 		}
// 	}
}

void xSmoothedParticleHydrodynamicsModel::ExportParticleDataForView(std::string& path)
{
	std::fstream of;
	of.open(path, std::ios::out | std::ios::binary);
	of.write((char*)&np, sizeof(unsigned int));
	of.write((char*)&nfluid, sizeof(unsigned int));
	of.write((char*)&nbound, sizeof(unsigned int));
	of.write((char*)&ndummy, sizeof(unsigned int));
	of.write((char*)&k_viscosity, sizeof(double));
	of.write((char*)&ref_rho, sizeof(double));
	of.write((char*)&fs_factor, sizeof(double));
	of.write((char*)&pspace, sizeof(double));
	of.write((char*)&water_depth, sizeof(double));
	of.write((char*)&corr, sizeof(int));
	of.write((char*)&turb, sizeof(int));
	of.write((char*)&bound, sizeof(int));
	of.write((char*)&ker_data, sizeof(xKernelFunctionData));
	of.write((char*)pos, sizeof(vector3d) * np);
	of.write((char*)vel, sizeof(vector3d) * np);
	of.write((char*)type, sizeof(xMaterialType) * np);
	of.close();
}

unsigned int xSmoothedParticleHydrodynamicsModel::NumTotalParticle()
{
	return np;
}

unsigned int xSmoothedParticleHydrodynamicsModel::NumFluid()
{
	return nfluid;
}

unsigned int xSmoothedParticleHydrodynamicsModel::NumBoundary()
{
	return nbound;
}

unsigned int xSmoothedParticleHydrodynamicsModel::NumDummy()
{
	return ndummy;
}

unsigned int xSmoothedParticleHydrodynamicsModel::Dimension()
{
	return dim;
}

double xSmoothedParticleHydrodynamicsModel::ParticleMass()
{
	return p_mass;
}

double xSmoothedParticleHydrodynamicsModel::ParticleVolume()
{
	return p_volume;
}

double xSmoothedParticleHydrodynamicsModel::ReferenceDensity()
{
	return ref_rho;
}

double xSmoothedParticleHydrodynamicsModel::ParticleSpacing()
{
	return pspace;
}

double xSmoothedParticleHydrodynamicsModel::KinematicViscosity()
{
	return k_viscosity;
}

double xSmoothedParticleHydrodynamicsModel::FreeSurfaceFactor()
{
	return fs_factor;
}

vector3d* xSmoothedParticleHydrodynamicsModel::Position()
{
	return pos;
}

vector3d* xSmoothedParticleHydrodynamicsModel::Velocity()
{
	return vel;
}

xKernelFunctionData& xSmoothedParticleHydrodynamicsModel::KernelData()
{
	return ker_data;
}

xWaveDampingData& xSmoothedParticleHydrodynamicsModel::WaveDampingData()
{
	return wave_damping_data;
}

unsigned int xSmoothedParticleHydrodynamicsModel::CreateOverlapCornerDummyParticle(unsigned int id, vector3d& p, vector3d& n1, vector3d& n2, bool isOnlyCount)
{
	unsigned int count = 0;
	//int layers = ;// (int)(fd->gridCellSize() / pspace) + 1;
	/*VEC3D v0 = vel;*/
	for (unsigned int i = 1; i <= nlayers; i++)
	{
		vector3d p1 = p - (i * pspace) * n1;
		vector3d p2 = p - (i * pspace) * n2;
		if (!isOnlyCount){
			double dist1 = length(p1 - p);
			double dist2 = length(p2 - p);
			vector3d norm1 = (p1 - p) / dist1;
			vector3d norm2 = (p2 - p) / dist2;
			vector3d p0 = (dist1 * norm1) + (dist2 * norm2) + p;
			pos[id] = p0;
			type[id] = DUMMY;//md->setParticle(wallId + count, DUMMY, 0.0, p0, v0);
		}
		count++;
// 		if (inner)
// 			continue;
		//count += 2;
		if (!isOnlyCount)
		{
			pos[id + count] = p1;// md->setParticle(wallId + count - 1, DUMMY, 0.0, p1, v0);
			pos[id + count + 1] = p2;// md->setParticle(wallId + count, DUMMY, 0.0, p2, v0);
			type[id + count] = DUMMY;
			type[id + count + 1] = DUMMY;
		}
		count += 2;

		if (i > 1){
			for (unsigned int j = 1; j < i; j++){
				//count += 2;
				vector3d p3 = p1 - (j * pspace) * n2;
				vector3d p4 = p2 - (j * pspace) * n1;
				if (!isOnlyCount){
					pos[id + count] = p3;
					pos[id + count + 1] = p4;
					type[id + count] = DUMMY;
					type[id + count + 1] = DUMMY;// 					md->setParticle(wallId + count - 1, DUMMY, 0.0, p3, v0);
					// 					md->setParticle(wallId + count, DUMMY, 0.0, p4, v0);
				}
				count += 2;
			}
		}
	}
	return count;
}

unsigned int xSmoothedParticleHydrodynamicsModel::CreateOverlapCornerDummyParticles(unsigned int overlap_sid, bool isOnlyCount)
{
	unsigned int count = 0;
	foreach(xOverlapCorner xoc, overlappingCorners)
	{
//		overlappingCorner *oc = &md->overlappingCorners[i];
		vector3d tan1 = new_vector3d(xoc.c1.tx, xoc.c1.ty, xoc.c1.tz);
		vector3d tan2 = new_vector3d(xoc.c2.tx, xoc.c2.ty, xoc.c2.tz);
		vector3d tv = tan1 - tan2;
		vector3d tu = tv / length(tv);
		if (!isOnlyCount){
			unsigned int id = overlap_sid + count;
			pos[id] = new_vector3d(xoc.c1.px, xoc.c1.py, xoc.c1.pz);
			type[id] = BOUNDARY;
			//particle* p = md->setParticle(id, BOUNDARY, 0.0, oc->c1.position, oc->iniVel);
			//
// 			p->setID(id);
// 			p->setIsCorner(true);
// 			p->setTangent(tu);
// 			p->setNormal(oc->c1.normal);
// 			p->setNormal2(oc->c2.normal);
			//				p->setAuxVelocity(oc->iniVel);
		}
		count++;
		//double dot = oc->c1.normal.dot(oc->c2.tangent);
		vector3d c1p = new_vector3d(xoc.c1.px, xoc.c1.py, xoc.c1.pz);
		vector3d c1n = new_vector3d(xoc.c1.nx, xoc.c1.ny, xoc.c1.nz);
		vector3d c2n = new_vector3d(xoc.c2.nx, xoc.c2.ny, xoc.c2.nz);
		if (/*dot <= 0 && */bound == DUMMY_PARTICLE_METHOD)
			count += CreateOverlapCornerDummyParticle(overlap_sid + count, c1p, c1n, c2n, isOnlyCount);
// 		else if (oc->inner && tboundary == DUMMY_PARTICLE_METHOD)
// 			count += 1 + createCornerDummyParticles(md->overlappingCornerStartIndex + count, oc->c1.position, oc->iniVel, oc->c1.normal, oc->c2.normal, isOnlyCount);
// 		else
// 			count += 1;
	}
	return count;
}
