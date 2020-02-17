#include "xXLSReader.h"
#include "xViewExporter.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
#include "xdynamics_manager/xSmoothedParticleHydrodynamicsModel.h"
#include "xdynamics_manager/xObjectManager.h"
#include "xdynamics_manager/xContactManager.h"
#include "xdynamics_simulation/xSimulation.h"
#include "ExcelFormat.h"

#include <sstream>

//using namespace ExcelFormat;
//static BasicExcel *xls;
//static XLSFormatManager *fmgr;
//static BasicExcelWorksheet* sheet;
typedef xUtilityFunctions uf;

xXLSReader::xXLSReader()
	: xve(NULL)
	, fmgr(NULL)
	, xls(NULL)
	, sheet(NULL)
{

}

xXLSReader::~xXLSReader()
{
	Release();
}

bool xXLSReader::_IsEmptyCell(int cid)
{
	/*if (cid == CELLTYPE_EMPTY || cid == CELLTYPE_BLANK || cid == CELLTYPE_ERROR)
		return true;*/
	return false;
}

bool xXLSReader::IsEmptyCell(int r, int c)
{
	YExcel::BasicExcelCell* cell = sheet->Cell(r, c);
	int b = cell->Type();
	//CellType cid = sheet->cellType(r, c);
	//return _IsEmptyCell(cid);
	return !b;
}

xPointMassData xXLSReader::ReadPointMassData(std::string& _name, int r, int& c, bool v)
{
	xPointMassData d = { 0, };
	double *ptr = &d.mass;
	d.mass = ReadNum(r, c++);
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 1);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 4);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 7);
	x = ReadStr(r, c++); x.split(",", 4, ptr + 10);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 14);

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
	xstring x;
	for (int i = 0; i < 5; i++)
	{
		x = ReadStr(r, c++);
		x.split(",", 3, ptr + i * 3);
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
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 6);
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
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 6);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 9);
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
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
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
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	d.minr = ReadNum(r, c++);
	d.maxr = ReadNum(r, c++);
	int t_shape = (int)CUBE_SHAPE;
	return d;
}

xPlaneParticleData xXLSReader::ReadPlaneParticleData(std::string & _name, int r, int & c)
{
	xPlaneParticleData d = { 0, };
	double *ptr = &d.dx;
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 6);
	d.minr = ReadNum(r, c++);
	d.maxr = ReadNum(r, c++);
	int t_shape = (int)PLANE_SHAPE;
	return d;
}

xLineParticleData xXLSReader::ReadLineParticleData(std::string & _name, int r, int & c)
{
	xLineParticleData d = { 0, };
	double* ptr = &d.sx;
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	d.minr = ReadNum(r, c++);
	d.maxr = ReadNum(r, c++);
	return d;
}

xListParticleData xXLSReader::ReadListParticleData(std::string& _name, int r, int& c)
{
	xListParticleData d = { 0 };
	unsigned int number = static_cast<int>(ReadNum(r, c++));
	int t_shape = (int)NO_SHAPE_AND_LIST;
	d.number = number;
	return d;
}

xCircleParticleData xXLSReader::ReadCircleParticleData(std::string& _name, int r, int& c)
{
	xCircleParticleData d = { 0, };
	double *ptr = &d.sx;
	xstring x;
	d.diameter = ReadNum(r, c++);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	d.minr = ReadNum(r, c++);
	d.maxr = ReadNum(r, c++);
	return d;
}

xContactParameterData xXLSReader::ReadContactData(std::string& _name, int r, int& c)
{
	xContactParameterData d = { 0, };
	d.rest = ReadNum(r, c++);
	d.rto = ReadNum(r, c++);
	d.mu_s = ReadNum(r, c++);
	d.mu = ReadNum(r, c++);
	d.coh = ReadNum(r, c++);
	d.rf = ReadNum(r, c++);
	return d;
}

xCylinderObjectData xXLSReader::ReadCylinderObjectData(std::string& _name, int mat, int r, int& c)
{
	xCylinderObjectData d = { 0, };
	d.length = ReadNum(r, c++);
	d.r_top = ReadNum(r, c++);
	d.r_bottom = ReadNum(r, c++);
	d.thickness = ReadNum(r, c++);
	double *ptr = &d.p0x;
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	x = ReadStr(r, c++);
	if (x == "top") d.empty = xCylinderObject::TOP_CIRCLE;
	else if (x == "bottom") d.empty = xCylinderObject::BOTTOM_CIRCLE;
	else if (x == "radial") d.empty = xCylinderObject::RADIAL_WALL;
	else if (x == "noempty") d.empty = xCylinderObject::NO_EMPTY_PART;
	else
	{
		std::string s;
		stringstream ss(s);
		ss << "Unsupported cylinder[" << _name << "] empty information.";
		throw runtime_error(ss.str().c_str());
	}
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
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	d.k = ReadNum(r, c++);
	d.c = ReadNum(r, c++);
	d.init_l = ReadNum(r, c++);
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

xRSDAData xXLSReader::ReadRSDAData(std::string& _name, int r, int& c)
{
	xRSDAData d = { 0, };
	double* ptr = &d.lx;
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 6);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 9);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 12);
	d.k = ReadNum(r, c++);
	d.c = ReadNum(r, c++);
	d.init_r = ReadNum(r, c++);
	if (xve)
	{
		int t = VRSDA;
		xve->Write((char*)&t, sizeof(int));
		unsigned int ns = static_cast<unsigned int>(_name.size());
		xve->Write((char*)&ns, sizeof(unsigned int));
		xve->Write((char*)_name.c_str(), sizeof(char)*ns);
		xve->Write((char*)&d, sizeof(xRSDAData));
	}
	return d;
}

xRotationalAxialForceData xXLSReader::ReadxRotationalAxialForceData(std::string& _name, int r, int& c)
{
	xRotationalAxialForceData d = { 0, };
	double* ptr = &d.lx;
	xstring x;
	x = ReadStr(r, c++); x.split(",", 3, ptr + 0);
	x = ReadStr(r, c++); x.split(",", 3, ptr + 3);
	d.rforce = ReadNum(r, c++);
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
// 	x = ReadStr(r, c++); uf::xsplit(x, ",", 2, ptr + 0);
// 	x = ReadStr(r, c++); uf::xsplit(x, ",", 3, ptr + 2);
// 	d.ps = ReadNum(r, c++);
// 	d.visc = ReadNum(r, c++);
// 	d.rho = ReadNum(r, c++);
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
	xls = new YExcel::BasicExcel(n);
	//fmgr = new ExcelFormat::XLSFormatManager(*xls);
	sheet = xls->GetWorksheet(0);

	return true;
}

void xXLSReader::Release()
{
	//if (book) book->release();
	//xls->Close();
	xls->Close();
	//if (sheet) delete sheet; sheet = NULL;
	//if (fmgr) delete fmgr; fmgr = NULL;
	if (xls) delete xls; xls = NULL;
	//xls.Close();
//	book->
//	book = NULL;
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
			std::string name = ReadStr(rc.x, rc.y++);
			if (xUtilityFunctions::ExistFile(name.c_str()))
			{
				xmbd->CreatePointMassesFromFile(name);			
			}				
			else
			{
				xObject *obj = xObjectManager::XOM()->XObject(name);
				xPointMassData xpmd = ReadPointMassData(name, rc.x, rc.y);
				if (!IsEmptyCell(rc.x, rc.y))
				{
					if (obj)
					{
						xPointMass* _xpm = dynamic_cast<xPointMass*>(obj);
						_xpm->setPosition(xpmd.px, xpmd.py, xpmd.pz);
						
					}
					rc.x++;
					continue;
				}
				xPointMass* xpm = NULL;
			
				xpm = xmbd->CreatePointMass(name);
				xpm->SetDataFromStructure(xmbd->NumMass(), xpmd);
 				if (obj)
 				{
 					obj->setDynamicsBody(true);
 					if (obj->Shape() == CYLINDER_SHAPE)
 					{
 						dynamic_cast<xCylinderObject*>(obj)->updateData();
 					}
 				}
			}			
			rc.x++;
			rc.y = init_col;
		}
	}
}

void xXLSReader::ReadJoint(xMultiBodyModel* xmbd, vector2i rc)
{
	while (1)
	{
		if (IsEmptyCell(rc.x, rc.y))	break;
		std::string name = ReadStr(rc.x, rc.y++);
		if (xUtilityFunctions::ExistFile(name.c_str()))
		{
			xmbd->CreateKinematicConstraintsFromFile(name);
		}
		else
		{
			std::cout << name << std::endl;
			xKinematicConstraint::cType type = (xKinematicConstraint::cType)static_cast<int>(ReadNum(rc.x, rc.y++));
			std::string base = ReadStr(rc.x, rc.y++);
			std::string action = ReadStr(rc.x, rc.y++);
			xKinematicConstraint* xkc = xmbd->CreateKinematicConstraint(name, type, base.c_str(), action.c_str());
			xkc->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadJointData(name, rc.x, rc.y));
			if (!IsEmptyCell(rc.x, rc.y))
			{
				int drc[2] = { rc.x, rc.y };// vector2i drc = rc;
				xstring str = ReadStr(drc[0], drc[1]++);
				str.split(",", 2, drc);
				drc[0] -= 1; drc[1] -= 1;
				if (!IsEmptyCell(drc[0], drc[1]))
				{
					xDrivingConstraint* xdc = NULL;
					name = ReadStr(drc[0], drc[1]++);
					xdc = xmbd->CreateDrivingConstraint(name, xkc);
					double stime = ReadNum(drc[0], drc[1]++);
					double cvel = ReadNum(drc[0]++, drc[1]++);
					xdc->setStartTime(stime);
					xdc->setConstantVelocity(cvel);
				}
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
		std::string name = ReadStr(rc.x, rc.y++);
		xForce::fType type = (xForce::fType)static_cast<int>(ReadNum(rc.x, rc.y++));
	
		std::string base = ReadStr(rc.x, rc.y++);
		std::string action = ReadStr(rc.x, rc.y++);
		
		if (type == xForce::TSDA_LIST_DATA)
		{
			xForce *xf = xdem->CreateForceElement(name, type, base, action);
			xTSDAData xt = { 0, };// ReadTSDAData(name, rc.x, rc.y);
			//xt.k = ReadNum(rc.x, rc.y++);
			//xt.c = ReadNum(rc.x, rc.y++);
			//xt.init_l = ReadNum(rc.x, rc.y++);
			std::string fpath = ReadStr(rc.x, rc.y);
			dynamic_cast<xSpringDamperForce*>(xf)->SetupDataFromListData(xt, fpath);
		}
		else if (type == xForce::RSDA_LIST_DATA) {
			xForce* xf = xdem->CreateForceElement(name, type, base, action);
			xRSDAData xt = { 0, };
			std::string fpath = ReadStr(rc.x, rc.y);
			dynamic_cast<xRotationSpringDamperForce*>(xf)->SetupDataFromListData(xt, fpath);
		}
		else
		{
			xForce* xf = xmbd->CreateForceElement(name, type, base, action);
			switch (xf->Type())
			{
			case xForce::TSDA: (dynamic_cast<xSpringDamperForce*>(xf))->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadTSDAData(name, rc.x, rc.y)); break;
			case xForce::RSDA: (dynamic_cast<xRotationSpringDamperForce*>(xf))->SetupDataFromStructure(xmbd->XMass(base), xmbd->XMass(action), ReadRSDAData(name, rc.x, rc.y)); break;
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
	d.type = static_cast<int>(ReadNum(rc.x, rc.y++));
	d.factor = ReadNum(rc.x, rc.y++);
	d.dim = static_cast<int>(ReadNum(rc.x, rc.y++));
	d.correction = static_cast<int>(ReadNum(rc.x, rc.y));
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
			std::string name = ReadStr(rc.x, rc.y++);
			xShapeType form = static_cast<xShapeType>(static_cast<int>(ReadNum(rc.x, rc.y++)));
			int material = -1;
			if (form != CLUSTER_SHAPE)
				material = static_cast<int>(ReadNum(rc.x, rc.y++));
			if (form == CUBE_SHAPE)
			{
				xCubeParticleData d = ReadCubeParticleData(name, rc.x, rc.y);
				unsigned int np = xdem->XParticleManager()->GetNumCubeParticles(d.dx, d.dy, d.dz, d.minr, d.maxr);
				xdem->XParticleManager()->CreateCubeParticle(name, (xMaterialType)material, np, d);			
			}
			else if (form == PLANE_SHAPE)
			{
				xPlaneParticleData d = ReadPlaneParticleData(name, rc.x, rc.y);
				xpo = xdem->XParticleManager()->CreatePlaneParticle(name, (xMaterialType)material, d);
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
				xpo = xdem->XParticleManager()->CreateCircleParticle(name.c_str(), (xMaterialType)material, npcircle, d);
				
			}
			else if (form == CLUSTER_SHAPE)
			{
				std::string obj;
				obj = ReadStr(rc.x, rc.y++);
				if (obj == "cmf") {
					material = static_cast<int>(ReadNum(rc.x, rc.y++));
					vector3d loc = new_vector3d(0, 0, 0);
					xstring x = ReadStr(rc.x, rc.y++);
					x.split(",", 3, &loc.x);
					xstring filepath = ReadStr(rc.x, rc.y++);
					xdem->XParticleManager()->SetClusterParticlesFromGenModel(name, (xMaterialType)material, loc, filepath.toStdString());
				}
				else {
					xObject* xo = xom->XObject(obj);
					vector3d loc = new_vector3d(0, 0, 0);
					vector3i grid = new_vector3i(0, 0, 0);
					xstring x = ReadStr(rc.x, rc.y++);
					x.split(",", 3, &loc.x);
					x = ReadStr(rc.x, rc.y++);
					x.split(",", 3, &grid.x);
					xstring israndom = ReadStr(rc.x, rc.y++);
					/*xUtilityFunctions::xsplit(ReadStr(rc.x, rc.y++), ",", 3, &loc.x);
					xUtilityFunctions::xsplit(ReadStr(rc.x, rc.y++), ",", 3, &grid.x);*/
					xdem->XParticleManager()->CreateClusterParticle(name.c_str(), xo->Material(), loc, grid, dynamic_cast<xClusterObject*>(xo), israndom == "true" ? true : false);
				}
			}
			else if (form == NO_SHAPE_AND_MASS)
			{
				double rad = ReadNum(rc.x, rc.y++);
				xPointMassData pm = ReadPointMassData(name, rc.x, rc.y, false);
				xdem->XParticleManager()->CreateMassParticle(name, (xMaterialType)material, rad, pm);
			}
			else if (form == NO_SHAPE_AND_LIST)
			{
				//xListParticleData d = ReadListParticleData(name, rc.x++, rc.y);
				unsigned int number = static_cast<int>(ReadNum(rc.x, rc.y++));
				std::string x;
				std::string fpath = ReadStr(rc.x, rc.y++);
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
				xstring x = ReadStr(rc.x, rc.y);
				std::string ext = xUtilityFunctions::FileExtension(x.text());
				if(ext == ".bin")
				{
					std::string p_path = ReadStr(rc.x, rc.y);
					xdem->XParticleManager()->SetChangeParticlesFilePath(p_path);
					//xdem->XParticleManager()->SetCurrentParticlesFromPartResult(p_path);
				}
				else
				{
					unsigned int np = 0;
					unsigned int neach = 0;
					unsigned int nstep = 0;
					vector2i _rc = new_vector2i(0, 0);

					if (x.size())// xUtilityFunctions::xsplit(ReadStr(rc.x, rc.y), ",", 2, &_rc.x))
					{
						x.split(",", 2, &_rc.x);
						unsigned int _neach = 0;
						_rc.x -= 1; _rc.y -= 1;
						if (!IsEmptyCell(_rc.x, _rc.y))
						{
							np = static_cast<unsigned int>(ReadNum(_rc.x, _rc.y++));
							neach = static_cast<unsigned int>(ReadNum(_rc.x, _rc.y++));
							nstep = static_cast<unsigned int>(ReadNum(_rc.x, _rc.y));
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
	//	std::string name = xUtilityFunctions::WideChar2String(ReadStr(rc.x, rc.y++));
	//	double loc[3] = { 0, };			
	double visc = 0;
	double rho = 0;
	double ps = 0;
	//uf::xsplit(ReadStr(rc.x, rc.y++), ",", 3, loc);
	ps = ReadNum(rc.x, rc.y++);
	visc = ReadNum(rc.x, rc.y++);
	rho = ReadNum(rc.x, rc.y++);
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
			std::string name = ReadStr(rc.x, rc.y++);
			xContactForceModelType method = static_cast<xContactForceModelType>(static_cast<int>(ReadNum(rc.x, rc.y++)));
			std::string obj0 = ReadStr(rc.x, rc.y++);
			std::string obj1 = ReadStr(rc.x, rc.y++);
			xContactParameterData d = ReadContactData(name, rc.x, rc.y);
		
			xObject* obj0_ptr = xObjectManager::XOM()->XObject(obj0);
			xObject* obj1_ptr = xObjectManager::XOM()->XObject(obj1);
			if (obj0_ptr == NULL)
				throw runtime_error(std::string("You have defined a contact[") + name + "] for an object[" + obj0 + "] that does not exist.");
			if (obj1_ptr == NULL)
				throw runtime_error(std::string("You have defined a contact[") + name + "] for an object[" + obj1 + "] that does not exist.");
			xcm->CreateContactPair(
				name,
				method,
				obj0_ptr,
				obj1_ptr,
				d);
				
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
			std::string name = ReadStr(rc.x, rc.y++);
			xShapeType form = static_cast<xShapeType>(static_cast<int>(ReadNum(rc.x, rc.y++)));
			int material = static_cast<int>(ReadNum(rc.x, rc.y++));
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
				unsigned int num = ReadNum(rc.x, rc.y++);
				//double min_rad = ReadNum(rc.x, rc.y++);
				double rad = ReadNum(rc.x, rc.y++);
				//unsigned int isEachCluster = ReadNum(rc.x, rc.y++);
				xstring sdata;
				int loc[2] = { 0, };
				sdata = ReadStr(rc.x, rc.y++); sdata.split(",", 2, loc);
				vector4d *d = new vector4d[num];
				loc[0] -= 1; loc[1] -= 1;
				for (unsigned int i = 0; i < num; i++)
				{
					double v[3] = { 0, };
					sdata = ReadStr(loc[0], loc[1]++);
					sdata.split(",", 3, v);
					d[i] = new_vector4d(v[0], v[1], v[2], rad);
				}
				xcs->setClusterSet(num, rad, rad, d, 0);
				delete[] d;
			}
			else if (form == xShapeType::MESH_SHAPE)
			{
				xMeshObject* xmo = xom->CreateMeshShapeObject(name, material);
				xstring x;
				vector3d loc;
				double fsz = 0.0;
				x = ReadStr(rc.x, rc.y++); x.split(",", 3, &(loc.x) + 0);
				fsz = ReadNum(rc.x, rc.y++);
				xmo->setRefinementSize(fsz);
				std::string mf = ReadStr(rc.x, rc.y++);
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
			else
			{
				std::string s;
				stringstream ss(s);
				ss << "Unsupported shape[" << name << "] type.";
				throw runtime_error(ss.str().c_str());
			}
			if (!IsEmptyCell(rc.x, rc.y))
			{
				double v[3] = { 0.0, };
				xstring ch_vel = ReadStr(rc.x, rc.y++);
				ch_vel.split(",", 3, v);
				xPointMass* xpm = xom->setMovingConstantMovingVelocity(name, v);
				xpm->setCompulsionMovingObject(true);
				if (!IsEmptyCell(rc.x, rc.y))
				{
					xstring ch;
					std::string och[3];
					ch = ReadStr(rc.x, rc.y++);
					ch.split(" ", 3, och);
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
		sol = ReadStr(rc.x, rc.y++);
		if (sol == "MBD")
		{
			xSimulation::MBDSolverType type = static_cast<xSimulation::MBDSolverType>(static_cast<int>(ReadNum(rc.x++, rc.y)));
			xSimulation::setMBDSolverType(type);
		}
		else if (sol == "DEM")
		{
			xSimulation::DEMSolverType type = static_cast<xSimulation::DEMSolverType>(static_cast<int>(ReadNum(rc.x++, rc.y)));
			xSimulation::setDEMSolverType(type);
		}
		else if (sol == "SPH")
		{
			xSimulation::SPHSolverType type = static_cast<xSimulation::SPHSolverType>(static_cast<int>(ReadNum(rc.x++, rc.y)));
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
// 		sol = ReadStr(rc.x, rc.y++);
// 		
// 	}
// 	rc.y = ic;
// 	if (!IsEmptyCell(rc.x, rc.y))
// 	{
// 		sol = ReadStr(rc.x, rc.y++);
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
		xSimulation::setTimeStep(ReadNum(rc.x, rc.y++));
	if (!IsEmptyCell(rc.x, rc.y))
		xSimulation::setSaveStep(static_cast<unsigned int>(ReadNum(rc.x, rc.y++)));
	if (!IsEmptyCell(rc.x, rc.y))
		xSimulation::setEndTime(ReadNum(rc.x, rc.y++));
}

void xXLSReader::ReadInputGravity(vector2i rc)
{
	if (!IsEmptyCell(rc.x, rc.y))
	{
		xstring x;
		double v3[3] = { 0, };
		x = ReadStr(rc.x, rc.y); x.split(",", 3, v3 + 0);
		xModel::setGravity(v3[0], v3[1], v3[2]);
	}
}

xstring xXLSReader::SetupSheet(int idx)
{
	//if (book)
	//{
	//	sheet = book->getSheet(idx);
	//	return sheet->name();
	//}
	return "";
}

std::string xXLSReader::ReadStr(int r, int c)
{
	YExcel::BasicExcelCell* cell = sheet->Cell(r, c);
	const char* s = cell->GetString();
	if (s == NULL)
	{
		std::string out;
		stringstream ss(out);
		ss << "Cell[" << r << ", " << c << "] is not string type.";
		throw runtime_error(ss.str().c_str());
	}
		
	//CellFormat fmt(*fmgr, cell);

	////			cout << " - xf_idx=" << cell->GetXFormatIdx();

	///*const Workbook::Font& font = fmt_mgr.get_font(fmt);
	//string font_name = stringFromSmallString(font.name_);
	//cout << "  font name: " << font_name;*/
	////fmt.
	//const wstring& fmt_string = fmt.get_format_string();
	////cout << "  format: " << narrow_string(fmt_string);

	////cell->SetFormat(fmt_general);
	//std::string s = xUtilityFunctions::WideChar2String(fmt_string.c_str());
	/*if (sheet)
		s = ReadStr(r, c);*/
	return std::string(s);
}

double xXLSReader::ReadNum(int r, int c)
{
	YExcel::BasicExcelCell* cell = sheet->Cell(r, c);
	double s = cell->GetDouble();
	if (s == FLT_MAX)
	{
		std::string out;
		stringstream ss(out);
		ss << "Cell[" << r << ", " << c << "] is not number type.";
		throw runtime_error(ss.str().c_str());
	}
	return  s;
}

void xXLSReader::setViewExporter(xViewExporter* _xve)
{
	xve = _xve;
}

