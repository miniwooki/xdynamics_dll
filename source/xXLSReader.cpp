#include "xXLSReader.h"
#include "xViewExporter.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_simulation/xSimulation.h"

typedef xUtilityFunctions uf;

xXLSReader::xXLSReader()
	: book(NULL)
	, sheet(NULL)
	, xve(NULL)
{

}

xXLSReader::~xXLSReader()
{
	if (book) book->release();
}

bool xXLSReader::_IsEmptyCell(int cid)
{
	if (cid == CELLTYPE_EMPTY || cid == CELLTYPE_BLANK || cid == CELLTYPE_ERROR)
		return true;
	return false;
}

bool xXLSReader::IsEmptyCell(int r, int c)
{
	CellType cid = sheet->cellType(r, c);
	return _IsEmptyCell(cid);
}

xPointMassData xXLSReader::ReadPointMassData(std::string& _name, int r, int& c, bool v)
{
	xPointMassData d = { 0, };
	double *ptr = &d.mass;
	d.mass = sheet->readNum(r, c++);
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 1);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 4);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 7);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 4, ptr + 10);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 14);
	if (xve && v)
	{
		int t = VMARKER;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size()); 
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xPointMassData));
	}
	return d;
}

xJointData xXLSReader::ReadJointData(std::string& _name, int r, int& c)
{
	xJointData d = { 0, };
	double *ptr = &d.lx;
	std::string x;
	for (int i = 0; i < 5; i++)
	{
		x = sheet->readStr(r, c++);
		uf::xsplit(x, ",", 3, ptr + i * 3);
	}
	if (xve)
	{
		int t = VJOINT;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size()); 
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xJointData));
	}
	return d;
}

xLineObjectData xXLSReader::ReadLineObjectData(std::string& _name, int mat, int r, int& c)
{
	xLineObjectData d = { 0, };
	double* ptr = &d.p0x;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 6);
	if (xve)
	{
		int t = VLINE;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size());
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xLineObjectData));
	}
	return d;
}

xPlaneObjectData xXLSReader::ReadPlaneObjectData(std::string& _name, int mat, int r, int& c)
{
	xPlaneObjectData d = { 0, };
	double *ptr = &d.p0x;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 6);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 9);
	if (xve)
	{
		int t = VPLANE;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size()); 
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xPlaneObjectData));
	}
	return d;
	
}

xCubeObjectData xXLSReader::ReadCubeObjectData(std::string& _name, int mat, int r, int& c)
{
	xCubeObjectData d = { 0, };
	double *ptr = &d.p0x;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	if (xve)
	{
		int t = VCUBE;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size()); 
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xCubeObjectData));
	}
	return d;
	
}

xCubeParticleData xXLSReader::ReadCubeParticleData(std::string& _name, int r, int& c)
{
	xCubeParticleData d = { 0, };
	double *ptr = &d.dx;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	d.minr = sheet->readNum(r, c++);
	d.maxr = sheet->readNum(r, c++);
	int t_shape = (int)CUBE_SHAPE;
// 	if (xve)
// 	{
// 		int t = VPARTICLE;
// 		xve->Write((char*)&t, sizeof(int));
// 		int ns = _name.size(); xve->Write((char*)&ns, sizeof(int));
// 		xve->Write((char*)_name.c_str(), sizeof(char)*_name.size());
// 		xve->Write((char*)t_shape, sizeof(int));
// 		xve->Write((char*)&d, sizeof(xCubeParticleData));
// 	}
	return d;
}

xLineParticleData xXLSReader::ReadLineParticleData(std::string & _name, int r, int & c)
{
	xLineParticleData d = { 0, };
	double* ptr = &d.sx;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	d.minr = sheet->readNum(r, c++);
	d.maxr = sheet->readNum(r, c++);
	return d;
}

xListParticleData xXLSReader::ReadListParticleData(std::string& _name, int r, int& c)
{
	xListParticleData d = { 0 };
	unsigned int number = static_cast<int>(sheet->readNum(r, c++));
	int t_shape = (int)NO_SHAPE_AND_LIST;
	d.number = number;
// 	if (xve)
// 	{
// 		int t = VPARTICLE;
// 		xve->Write((char*)&t, sizeof(int));
// 		int ns = _name.size(); xve->Write((char*)&ns, sizeof(int));
// 		xve->Write((char*)_name.c_str(), sizeof(char)*_name.size());
// 		xve->Write((char*)t_shape, sizeof(int));
// 		xve->Write((char*)&d, sizeof(xListParticleData));
// 	}
	return d;
}

xCircleParticleData xXLSReader::ReadCircleParticleData(std::string& _name, int r, int& c)
{
	xCircleParticleData d = { 0, };
	double *ptr = &d.sx;
	std::string x;
	d.diameter = sheet->readNum(r, c++);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	d.minr = sheet->readNum(r, c++);
	d.maxr = sheet->readNum(r, c++);
	return d;
}

xContactParameterData xXLSReader::ReadContactData(std::string& _name, int r, int& c)
{
	xContactParameterData d = { 0, };
	d.rest = sheet->readNum(r, c++);
	d.rto = sheet->readNum(r, c++);
	d.mu = sheet->readNum(r, c++);
	d.coh = sheet->readNum(r, c++);
	d.rf = sheet->readNum(r, c++);
	return d;
}

xCylinderObjectData xXLSReader::ReadCylinderObjectData(std::string& _name, int mat, int r, int& c)
{
	xCylinderObjectData d = { 0, };
	d.length = sheet->readNum(r, c++);
	d.r_top = sheet->readNum(r, c++);
	d.r_bottom = sheet->readNum(r, c++);
	d.thickness = sheet->readNum(r, c++);
	double *ptr = &d.p0x;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	x = sheet->readStr(r, c++);
	if (x == "top") d.empty = xCylinderObject::TOP_CIRCLE;
	else if (x == "bottom") d.empty = xCylinderObject::BOTTOM_CIRCLE;
	else if (x == "radial") d.empty = xCylinderObject::RADIAL_WALL;
	if (xve)
	{
		int t = VCYLINDER;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size());
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xCylinderObjectData));
	}
	return d;
}

xTSDAData xXLSReader::ReadTSDAData(std::string& _name, int r, int& c)
{
	xTSDAData d = { 0, };
	double* ptr = &d.spix;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	d.k = sheet->readNum(r, c++);
	d.c = sheet->readNum(r, c++);
	d.init_l = sheet->readNum(r, c++);
	if (xve)
	{
		int t = VTSDA;		
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size()); 
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xTSDAData));
	}
	return d;
}

xRotationalAxialForceData xXLSReader::ReadxRotationalAxialForceData(std::string& _name, int r, int& c)
{
	xRotationalAxialForceData d = { 0, };
	double* ptr = &d.lx;
	std::string x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 3);
	d.rforce = sheet->readNum(r, c++);
	if (xve)
	{
		int t = VRAXIAL;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size());
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xRotationalAxialForceData));
	}
	return d;
}

// xSPHPlaneObjectData xXLSReader::ReadSPHPlaneParticleData(std::string& _name, int r, int& c)
// {
// 	xSPHPlaneObjectData d = { 0, };
// 	double* ptr = &d.dx;
// 	std::string x;
// 	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 2, ptr + 0);
// 	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 2);
// 	d.ps = sheet->readNum(r, c++);
// 	d.visc = sheet->readNum(r, c++);
// 	d.rho = sheet->readNum(r, c++);
// 	if (xve)
// 	{
// 		int t = VSPHPLANE;
// 		xve->Write((char*)&t, sizeof(int));
// 		unsigned int ns = static_cast<unsigned int>(_name.size());
// 		xve->Write((char*)&ns, sizeof(unsigned int));
// 		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
// 		xve->Write((char*)&d, sizeof(xSPHPlaneObjectData));
// 	}
// }

bool xXLSReader::Load(const char* n)
{
	book = xlCreateBook();
	if (book)
	{
		connect_file = "ddfafa";
		return book->load(n);
	}
	return false;
}

void xXLSReader::ReadMass(xMultiBodyModel* xmbd, vector2i rc)
{
	if (xmbd)
	{
		int init_col = rc.y;
		//std::string name, str;			
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y)) break;
			std::string name = sheet->readStr(rc.x, rc.y++);
			xObject *obj = xObjectManager::XOM()->XObject(name);
			xPointMass* xpm = NULL;
			xpm = xmbd->CreatePointMass(name);
			if (obj) obj->setMovingObject(true);
			xPointMassData xpmd = ReadPointMassData(name, rc.x++, rc.y);
			xpm->SetDataFromStructure(xmbd->NumMass(), xpmd);
// 			if (xpm->Shape() == MESH_SHAPE)
// 				dynamic_cast<xMeshObject*>(xpm)->translation(new_vector3d(xpmd.px, xpmd.py, xpmd.pz));
			
			
			//xpm->translation(xpm->Position());
			// obj->TranslationPosition(xpm->Position());
			rc.y = init_col;
		}
	}
}

void xXLSReader::ReadJoint(xMultiBodyModel* xmbd, vector2i rc)
{
	while (1)
	{
		if (IsEmptyCell(rc.x, rc.y))	break;
		std::string name = sheet->readStr(rc.x, rc.y++);
		xKinematicConstraint::cType type = (xKinematicConstraint::cType)static_cast<int>(sheet->readNum(rc.x, rc.y++));
		std::string base = sheet->readStr(rc.x, rc.y++);
		std::string action = sheet->readStr(rc.x, rc.y++);
		xKinematicConstraint* xkc = xmbd->CreateKinematicConstraint(name, type, base.c_str(), action.c_str());
		xkc->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadJointData(name, rc.x, rc.y));
		if (!IsEmptyCell(rc.x, rc.y))
		{
			vector2i drc = rc;
			std::string str = sheet->readStr(drc.x, drc.y++);
			uf::xsplit(str, ",", drc);
			drc.x -= 1; drc.y -= 1;
			if (!IsEmptyCell(drc.x, drc.y))
			{
				xDrivingConstraint* xdc = NULL;
				name = sheet->readStr(drc.x, drc.y++);
				xdc = xmbd->CreateDrivingConstraint(name, xkc);
				double stime = sheet->readNum(drc.x, drc.y++);
				double cvel = sheet->readNum(drc.x++, drc.y++);
				xdc->setStartTime(stime);
				xdc->setConstantVelocity(cvel);
			}
		}
		rc.x++;
		rc.y = 0;
	}
}

void xXLSReader::ReadForce(xMultiBodyModel* xmbd, xDiscreteElementMethodModel* xdem, vector2i rc)
{
	vector2i init_rc = rc;
	while (1)
	{
		if (IsEmptyCell(rc.x, rc.y)) break;
		std::string name = sheet->readStr(rc.x, rc.y++);
		xForce::fType type = (xForce::fType)static_cast<int>(sheet->readNum(rc.x, rc.y++));
	
		std::string base = sheet->readStr(rc.x, rc.y++);
		std::string action = sheet->readStr(rc.x, rc.y++);
		
		if (type == xForce::TSDA_LIST_DATA)
		{
			xSpringDamperForce *xf = xdem->CreateForceElement(name, type, base, action);
			xTSDAData xt = { 0, };// ReadTSDAData(name, rc.x, rc.y);
			xt.k = sheet->readNum(rc.x, rc.y++);
			xt.c = sheet->readNum(rc.x, rc.y++);
			xt.init_l = sheet->readNum(rc.x, rc.y++);
			std::string fpath = sheet->readStr(rc.x, rc.y);
			xf->SetupDataFromListData(xt, fpath);
		}
		else
		{
			xForce* xf = xmbd->CreateForceElement(name, type, base, action);
			switch (xf->Type())
			{
			case xForce::TSDA: (dynamic_cast<xSpringDamperForce*>(xf))->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadTSDAData(name, rc.x, rc.y)); break;
			case xForce::RSDA: break;
			case xForce::RAXIAL: (dynamic_cast<xRotationalAxialForce*>(xf))->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadxRotationalAxialForceData(name, rc.x, rc.y)); break;
			}
		}
			
		//xkc->SetupDataFromStructure(ReadJointData(rc.x, rc.y));
		

		rc.x++;
		rc.y = init_rc.y;
	}
}

void xXLSReader::ReadKernel(xSmoothedParticleHydrodynamicsModel* xsph, vector2i rc)
{
	if (IsEmptyCell(rc.x, rc.y)) return;
	xKernelFunctionData d = { 0, };
	d.type = static_cast<int>(sheet->readNum(rc.x, rc.y++));
	d.factor = sheet->readNum(rc.x, rc.y++);
	d.dim = static_cast<int>(sheet->readNum(rc.x, rc.y++));
	d.correction = static_cast<int>(sheet->readNum(rc.x, rc.y));
	xsph->setKernelFunctionData(d);
}

void xXLSReader::ReadDEMParticle(xDiscreteElementMethodModel* xdem, xObjectManager* xom, vector2i rc)
{
	if (xdem->XParticleManager())
	{
		int init_col = rc.y;
		xParticleObject* xpo = NULL;
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y))
				break;
			std::string name = sheet->readStr(rc.x, rc.y++);
			xShapeType form = static_cast<xShapeType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			int material = -1;
			if (form != CLUSTER_SHAPE)
				material = static_cast<int>(sheet->readNum(rc.x, rc.y++));
			if (form == CUBE_SHAPE)
			{
				xCubeParticleData d = ReadCubeParticleData(name, rc.x, rc.y);
				unsigned int np = xdem->XParticleManager()->GetNumCubeParticles(d.dx, d.dy, d.dz, d.minr, d.maxr);
				xdem->XParticleManager()->CreateCubeParticle(name, (xMaterialType)material, np, d);			
			}
			else if (form == PLANE_SHAPE)
			{

			}
			else if (form == LINE_SHAPE)
			{
				xLineParticleData d = ReadLineParticleData(name, rc.x, rc.y);
				unsigned int np = xdem->XParticleManager()->GetNumLineParticles(d.sx, d.sy, d.sz, d.ex, d.ey, d.ez, d.minr, d.maxr);
				xdem->XParticleManager()->CreateLineParticle(name, (xMaterialType)material, np, d);
			}
			else if (form == CIRCLE_SHAPE)
			{
				xCircleParticleData d = ReadCircleParticleData(name, rc.x, rc.y);
				unsigned int p_np = xdem->XParticleManager()->NumParticle();
			
				unsigned int npcircle = xdem->XParticleManager()->GetNumCircleParticles(d.diameter, d.minr, d.maxr);
				/*if (!IsEmptyCell(rc.x, rc.y))
				{
					vector2i _rc = new_vector2i(0, 0);
					xUtilityFunctions::xsplit(sheet->readStr(rc.x, rc.y++), ",", 2, &_rc.x);
					unsigned int _neach = 0;
					_rc.x -= 1; _rc.y -= 1;
					if (!IsEmptyCell(_rc.x, _rc.y))
					{
						np = sheet->readNum(_rc.x, _rc.y++);
						neach = sheet->readNum(_rc.x, _rc.y++);
						nstep = sheet->readNum(_rc.x, _rc.y);
					}
					if (!neach)
						neach = npcircle;
				}
				else
					np = xdem->XParticleManager()->GetNumCircleParticles(d.diameter, d.minr, d.maxr);*/
				xpo = xdem->XParticleManager()->CreateCircleParticle(name.c_str(), (xMaterialType)material, npcircle, d);
				
			}
			else if (form == CLUSTER_SHAPE)
			{
				std::string obj;
				obj = sheet->readStr(rc.x, rc.y++);
				xObject* xo = xom->XObject(obj);

				/*unsigned int num = sheet->readNum(rc.x, rc.y++);*/
				vector3d loc = new_vector3d(0, 0, 0);
				vector3i grid = new_vector3i(0, 0, 0);
				xUtilityFunctions::xsplit(sheet->readStr(rc.x, rc.y++), ",", 3, &loc.x);
				xUtilityFunctions::xsplit(sheet->readStr(rc.x, rc.y++), ",", 3, &grid.x);
				xdem->XParticleManager()->CreateClusterParticle(name.c_str(), xo->Material(), loc, grid, dynamic_cast<xClusterObject*>(xo));
			}
			else if (form == NO_SHAPE_AND_MASS)
			{
				double rad = sheet->readNum(rc.x, rc.y++);
				xPointMassData pm = ReadPointMassData(name, rc.x, rc.y, false);
				xdem->XParticleManager()->CreateMassParticle(name, (xMaterialType)material, rad, pm);
			}
			else if (form == NO_SHAPE_AND_LIST)
			{
				//xListParticleData d = ReadListParticleData(name, rc.x++, rc.y);
				unsigned int number = static_cast<int>(sheet->readNum(rc.x, rc.y++));
				std::string x;
				std::string fpath = sheet->readStr(rc.x, rc.y++);
				vector4d* d = new vector4d[number];
				double* m = new double[number];
				std::fstream fs;
				fs.open(fpath.c_str(), std::ios::in);
				if (fs.is_open())
				{
					for (unsigned int i = 0; i < number; i++)
						fs >> d[i].x >> d[i].y >> d[i].z >> d[i].w >> m[i];
				}
				else
				{
					std::cout << fpath << " not open" << std::endl;
				}
				
				fs.close();
				xdem->XParticleManager()->CreateParticleFromList(name.c_str(), (xMaterialType)material, number, d, m);
				delete[] d;
				delete[] m;
			}
			
			if (!IsEmptyCell(rc.x, rc.y))
			{
				unsigned int np = 0;
				unsigned int neach = 0;
				unsigned int nstep = 0;
				vector2i _rc = new_vector2i(0, 0);
				if (xUtilityFunctions::xsplit(sheet->readStr(rc.x, rc.y), ",", 2, &_rc.x))
				{
					unsigned int _neach = 0;
					_rc.x -= 1; _rc.y -= 1;
					if (!IsEmptyCell(_rc.x, _rc.y))
					{
						np = sheet->readNum(_rc.x, _rc.y++);
						neach = sheet->readNum(_rc.x, _rc.y++);
						nstep = sheet->readNum(_rc.x, _rc.y);
					}
					/*		if (!neach)
								neach = npcircle;*/
					if (neach && nstep)
					{
						xpo->setEachCount(xpo->NumParticle());
						xParticleCreateCondition xpcc = { xpo->StartIndex(), np, neach, nstep };
						xdem->XParticleManager()->AddParticleCreatingCondition(xpo, xpcc);
					}
					rc.y++;
				}				
			}
			if (!IsEmptyCell(rc.x, rc.y))
			{
				std::string p_path = sheet->readStr(rc.x, rc.y);
				xdem->XParticleManager()->SetCurrentParticlesFromPartResult(p_path);
			}
			rc.x++;
			rc.y = init_col;
		}
	}
}

void xXLSReader::ReadSPHParticle(xSmoothedParticleHydrodynamicsModel* xsph, vector2i rc)
{
	if (IsEmptyCell(rc.x, rc.y))
		return;
	//	std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
	//	double loc[3] = { 0, };			
	double visc = 0;
	double rho = 0;
	double ps = 0;
	//uf::xsplit(sheet->readStr(rc.x, rc.y++), ",", 3, loc);
	ps = sheet->readNum(rc.x, rc.y++);
	visc = sheet->readNum(rc.x, rc.y++);
	rho = sheet->readNum(rc.x, rc.y++);
	//	xObject* xobj = xom->XObject(name);
	xsph->setParticleSpacing(ps);
	xsph->setKinematicViscosity(visc);
	xsph->setReferenceDensity(rho);
		//	xsph->XParticleManager()->CreateSPHParticles(xobj, ps);
}

void xXLSReader::ReadContact(xContactManager* xcm, vector2i rc)
{
	if (xcm)
	{
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y))
				break;
			std::string name = sheet->readStr(rc.x, rc.y++);
			xContactForceModelType method = static_cast<xContactForceModelType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			std::string obj0 = sheet->readStr(rc.x, rc.y++);
			std::string obj1 = sheet->readStr(rc.x, rc.y++);
			xContactParameterData d = ReadContactData(name, rc.x, rc.y);
			xContact* xc = xcm->CreateContactPair(
				name,
				method,
				xObjectManager::XOM()->XObject(obj0),
				xObjectManager::XOM()->XObject(obj1),
				d);
		/*	if (!IsEmptyCell(rc.x, rc.y))
			{
				double mul = sheet->readNum(rc.x, rc.y);
				xc->setStiffMultiplyer(mul);
			}*/
			rc.x++;
			rc.y = 0;
		}
	}
}

void xXLSReader::ReadShapeObject(xObjectManager* xom, vector2i rc)
{
	if (xom)
	{
		int init_col = rc.y;
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y)) break;
			std::string name = sheet->readStr(rc.x, rc.y++);
			xShapeType form = static_cast<xShapeType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			int material = static_cast<int>(sheet->readNum(rc.x, rc.y++));
			if (form == xShapeType::CUBE_SHAPE)
			{
				xCubeObject* xco = xom->CreateCubeShapeObject(name, material);
				xco->SetupDataFromStructure(ReadCubeObjectData(name, material, rc.x, rc.y));
			}
			else if (form == xShapeType::LINE_SHAPE)
			{
				xLineObject* xlo = xom->CreateLineShapeObject(name, material);
				xlo->SetupDataFromStructure(ReadLineObjectData(name, material, rc.x, rc.y));
			}
			else if (form == xShapeType::PLANE_SHAPE)
			{
				xPlaneObject* xpo = xom->CreatePlaneShapeObject(name, material);
				xpo->SetupDataFromStructure(ReadPlaneObjectData(name, material, rc.x, rc.y));
			}
			else if (form == xShapeType::CYLINDER_SHAPE)
			{
				xCylinderObject* xco = xom->CreateCylinderShapeObject(name, material);
				xco->SetupDataFromStructure(ReadCylinderObjectData(name, material, rc.x, rc.y));
			}
			else if (form == xShapeType::CLUSTER_SHAPE)
			{
				xClusterObject* xcs = xom->CreateClusterShapeObject(name, material);
				unsigned int num = sheet->readNum(rc.x, rc.y++);
				double rad = sheet->readNum(rc.x, rc.y++);
				std::string sdata;
				int loc[2] = { 0, };
				sdata = sheet->readStr(rc.x, rc.y++); uf::xsplit(sdata, ",", 2, loc);
				vector3d *d = new vector3d[num];
				loc[0] -= 1; loc[1] -= 1;
				for (unsigned int i = 0; i < num; i++)
				{
					double v[3] = { 0, };
					sdata = sheet->readStr(loc[0], loc[1]++);
					uf::xsplit(sdata, ",", 3, v);
					d[i] = new_vector3d(v[0], v[1], v[2]);
				}
				xcs->setClusterSet(num, rad, d);
				delete[] d;
			}
			else if (form == xShapeType::MESH_SHAPE)
			{
				xMeshObject* xmo = xom->CreateMeshShapeObject(name, material);
				std::string x;
				vector3d loc;
				double fsz = 0.0;
				x = sheet->readStr(rc.x, rc.y++); uf::xsplit(x, ",", 3, &(loc.x) + 0);
				fsz = sheet->readNum(rc.x, rc.y++);
				xmo->setRefinementSize(fsz);
				std::string mf = sheet->readStr(rc.x, rc.y++);
				xmo->DefineShapeFromFile(loc, mf);
				std::string file = xModel::makeFilePath(name + ".mesh");
				int t = VMESH;
				xve->Write((char*)&t, sizeof(int));
				unsigned int ns = (unsigned int)file.size();
				xve->Write((char*)&ns, sizeof(unsigned int));
				xve->Write((char*)file.c_str(), sizeof(char)*ns);
				xmo->exportMeshData(file);
				//xmo->splitTriangles(fsz);
			}
			if (!IsEmptyCell(rc.x, rc.y))
			{
				double v = 0.0;
				v = sheet->readNum(rc.x, rc.y++);
				xPointMass* xpm = xom->setMovingConstantMovingVelocity(name, v);
				if (!IsEmptyCell(rc.x, rc.y))
				{
					std::string ch;
					std::string och[3];
					ch = sheet->readStr(rc.x, rc.y++);
					xUtilityFunctions::xsplit(ch.c_str(), " ", 3, och);
					double stop_value = atof(och[2].c_str());
					xSimulationStopType xssc;
					xComparisonType xct;
					if (och[0] == "FM")	{ xssc = FORCE_MAGNITUDE; }
					if (och[1] == ">") { xct = GRATER_THAN; }
					xpm->setStopCondition(xssc, xct, stop_value);
				}
			}
			rc.x++;
			rc.y = init_col;
		}
	}
}

void xXLSReader::ReadIntegrator(vector2i rc)
{
	std::string sol;
	int ic = rc.y;
	while (!IsEmptyCell(rc.x, rc.y))
	{
		sol = sheet->readStr(rc.x, rc.y++);
		if (sol == "MBD")
		{
			xSimulation::MBDSolverType type = static_cast<xSimulation::MBDSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setMBDSolverType(type);
		}
		else if (sol == "DEM")
		{
			xSimulation::DEMSolverType type = static_cast<xSimulation::DEMSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setDEMSolverType(type);
		}
		else if (sol == "SPH")
		{
			xSimulation::SPHSolverType type = static_cast<xSimulation::SPHSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setSPHSolverType(type);
		}
		//rc.x++;
		rc.y = ic;
	}
	
		
// 		
// 	}
// 	rc.y = ic;
// 	if (!IsEmptyCell(rc.x, rc.y))
// 	{
// 		sol = sheet->readStr(rc.x, rc.y++);
// 		
// 	}
// 	rc.y = ic;
// 	if (!IsEmptyCell(rc.x, rc.y))
// 	{
// 		sol = sheet->readStr(rc.x, rc.y++);
// 		
// 	}
}

void xXLSReader::ReadSimulationCondition(vector2i rc)
{
	if (!IsEmptyCell(rc.x, rc.y))
	{
		xstring dev = ReadStr(rc.x, rc.y++);
		if (dev == "CPU") xSimulation::setCPUDevice();
		else if (dev == "GPU") xSimulation::setGPUDevice();
		else xLog::log("Device input error : You have entered an invalid device type.(" + dev.toStdString() + ")");
	}
	if (!IsEmptyCell(rc.x, rc.y))
		xSimulation::setTimeStep(sheet->readNum(rc.x, rc.y++));
	if (!IsEmptyCell(rc.x, rc.y))
		xSimulation::setSaveStep(static_cast<unsigned int>(sheet->readNum(rc.x, rc.y++)));
	if (!IsEmptyCell(rc.x, rc.y))
		xSimulation::setEndTime(sheet->readNum(rc.x, rc.y++));
}

void xXLSReader::ReadInputGravity(vector2i rc)
{
	if (!IsEmptyCell(rc.x, rc.y))
	{
		std::string x;
		double v3[3] = { 0, };
		x = sheet->readStr(rc.x, rc.y); uf::xsplit(x, ",", 3, v3 + 0);
		xModel::setGravity(v3[0], v3[1], v3[2]);
	}
}

xstring xXLSReader::SetupSheet(int idx)
{
	if (book)
	{
		sheet = book->getSheet(idx);
		return sheet->name();
	}
	return "";
}

xstring xXLSReader::ReadStr(int r, int c)
{
	std::string s;
	if (sheet)
		s = sheet->readStr(r, c);
	return s;
}

void xXLSReader::setViewExporter(xViewExporter* _xve)
{
	xve = _xve;
}

