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

xPointMassData xXLSReader::ReadPointMassData(std::string& _name, int r, int& c)
{
	xPointMassData d = { 0, };
	double *ptr = &d.mass;
	d.mass = sheet->readNum(r, c++);
	std::wstring x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 1);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 4);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 7);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 4, ptr + 10);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 14);
	if (xve)
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
	std::wstring x;
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
	std::wstring x;
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
	std::wstring x;
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
	std::wstring x;
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
	std::wstring x;
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

xTSDAData xXLSReader::ReadTSDAData(std::string& _name, int r, int& c)
{
	xTSDAData d = { 0, };
	double* ptr = &d.spix;
	std::wstring x;
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
	std::wstring x;
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
// 	std::wstring x;
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

bool xXLSReader::Load(const wchar_t* n)
{
	book = xlCreateBook();
	if (book)
	{
		
		return book->load(n);
	}
	return false;
}

void xXLSReader::ReadMass(xMultiBodyModel* xmbd, vector2i rc)
{
	if (xmbd)
	{
		int init_col = rc.y;
		//std::wstring name, str;			
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y)) break;
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xObject *obj = xObjectManager::XOM()->XObject(name);
			xPointMass* xpm = NULL;
			xpm = xmbd->CreatePointMass(name);

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
		std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
		xKinematicConstraint::cType type = (xKinematicConstraint::cType)static_cast<int>(sheet->readNum(rc.x, rc.y++));
		std::wstring base = sheet->readStr(rc.x, rc.y++);
		std::wstring action = sheet->readStr(rc.x, rc.y++);
		xKinematicConstraint* xkc = xmbd->CreateKinematicConstraint(name, type, uf::WideChar2String(base.c_str()), uf::WideChar2String(action.c_str()));
		xkc->SetupDataFromStructure(xmbd->XMass(xUtilityFunctions::xstring(base)), xmbd->XMass(xUtilityFunctions::xstring(action)), ReadJointData(name, rc.x, rc.y));
		if (!IsEmptyCell(rc.x, rc.y))
		{
			vector2i drc = rc;
			std::wstring str = sheet->readStr(drc.x, drc.y++);
			uf::xsplit(str, ",", drc);
			drc.x -= 1; drc.y -= 1;
			if (!IsEmptyCell(drc.x, drc.y))
			{
				xDrivingConstraint* xdc = NULL;
				name = xUtilityFunctions::WideChar2String(sheet->readStr(drc.x, drc.y++));
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

void xXLSReader::ReadForce(xMultiBodyModel* xmbd, vector2i rc)
{
	vector2i init_rc = rc;
	while (1)
	{
		if (IsEmptyCell(rc.x, rc.y)) break;
		std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
		xForce::fType type = (xForce::fType)static_cast<int>(sheet->readNum(rc.x, rc.y++));
		std::wstring base = sheet->readStr(rc.x, rc.y++);
		std::wstring action = sheet->readStr(rc.x, rc.y++);
		xForce* xf = xmbd->CreateForceElement(name, type, uf::WideChar2String(base.c_str()), uf::WideChar2String(action.c_str()));
		//xkc->SetupDataFromStructure(ReadJointData(rc.x, rc.y));
		switch (xf->Type())
		{
		case xForce::TSDA: (dynamic_cast<xSpringDamperForce*>(xf))->SetupDataFromStructure(xmbd->XMass(xUtilityFunctions::xstring(base)), xmbd->XMass(xUtilityFunctions::xstring(action)), ReadTSDAData(name, rc.x, rc.y)); break;
		case xForce::RSDA: break;
		case xForce::RAXIAL: (dynamic_cast<xRotationalAxialForce*>(xf))->SetupDataFromStructure(xmbd->XMass(xUtilityFunctions::xstring(base)), xmbd->XMass(xUtilityFunctions::xstring(action)), ReadxRotationalAxialForceData(name, rc.x, rc.y)); break;
		}
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

void xXLSReader::ReadDEMParticle(xDiscreteElementMethodModel* xdem, vector2i rc)
{
	if (xdem->XParticleManager())
	{
		int init_col = rc.y;
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y))
				break;
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xShapeType form = static_cast<xShapeType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			int material = static_cast<int>(sheet->readNum(rc.x, rc.y++));
			if (form == CUBE_SHAPE)
			{
				xCubeParticleData d = ReadCubeParticleData(name, rc.x, rc.y);
				unsigned int np = xdem->XParticleManager()->GetNumCubeParticles(d.dx, d.dy, d.dz, d.minr, d.maxr);
				xdem->XParticleManager()->CreateCubeParticle(name.c_str(), (xMaterialType)material, np, d);			
			}
			else if (form == PLANE_SHAPE)
			{

			}

			if (form == NO_SHAPE_AND_LIST)
			{
				//xListParticleData d = ReadListParticleData(name, rc.x++, rc.y);
				unsigned int number = static_cast<int>(sheet->readNum(rc.x, rc.y++));
				std::wstring x;
				x = sheet->readStr(rc.x++, rc.y); 
				vector2i _rc;
				uf::xsplit(x, ",", 2, &_rc.x);
				vector4d* d = new vector4d[number];
				_rc.x -= 1; _rc.y -= 1;
				int start_column = _rc.y;
				for (unsigned int i = 0; i < number; i++)
				{
					if (IsEmptyCell(_rc.x, _rc.y))
						break;
					d[i].x = sheet->readNum(_rc.x, _rc.y++);
					d[i].y = sheet->readNum(_rc.x, _rc.y++);
					d[i].z = sheet->readNum(_rc.x, _rc.y++);
					d[i].w = sheet->readNum(_rc.x++, _rc.y);
					_rc.y = start_column;
				}
				xdem->XParticleManager()->CreateParticleFromList(name.c_str(), (xMaterialType)material, number, d);
				delete[] d;
			}
			if (!IsEmptyCell(rc.x, rc.y))
			{
				std::string p_path = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y));
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
			std::string name = uf::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xContactForceModelType method = static_cast<xContactForceModelType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			std::wstring obj0 = sheet->readStr(rc.x, rc.y++);
			std::wstring obj1 = sheet->readStr(rc.x, rc.y++);
			xContactParameterData d = ReadContactData(name, rc.x++, rc.y);
			xContact* xc = xcm->CreateContactPair(
				name,
				method,
				xObjectManager::XOM()->XObject(uf::WideChar2String(obj0.c_str())),
				xObjectManager::XOM()->XObject(uf::WideChar2String(obj1.c_str())),
				d);
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
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
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
			else if (form == xShapeType::MESH_SHAPE)
			{
				xMeshObject* xmo = xom->CreateMeshShapeObject(name, material);
				std::wstring x;
				vector3d loc;
				double fsz = 0.0;
				x = sheet->readStr(rc.x, rc.y++); uf::xsplit(x, ",", 3, &(loc.x) + 0);
				fsz = sheet->readNum(rc.x, rc.y++);
				xmo->setRefinementSize(fsz);
				std::string mf = uf::WideChar2String(sheet->readStr(rc.x, rc.y++));
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
			rc.x++;
			rc.y = init_col;
		}
	}
}

void xXLSReader::ReadIntegrator(vector2i rc)
{
	std::wstring sol;
	int ic = rc.y;
	while (!IsEmptyCell(rc.x, rc.y))
	{
		sol = sheet->readStr(rc.x, rc.y++);
		if (sol == L"MBD")
		{
			xSimulation::MBDSolverType type = static_cast<xSimulation::MBDSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setMBDSolverType(type);
		}
		else if (sol == L"DEM")
		{
			xSimulation::DEMSolverType type = static_cast<xSimulation::DEMSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setDEMSolverType(type);
		}
		else if (sol == L"SPH")
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
		QString dev = ReadStr(rc.x, rc.y++);
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
		std::wstring x;
		double v3[3] = { 0, };
		x = sheet->readStr(rc.x, rc.y); uf::xsplit(x, ",", 3, v3 + 0);
		xModel::setGravity(v3[0], v3[1], v3[2]);
	}
}

QString xXLSReader::SetupSheet(int idx)
{
	if (book)
	{
		sheet = book->getSheet(idx);
		return toWideCharToQString(sheet->name());
	}
	return "";
}

QString xXLSReader::ReadStr(int r, int c)
{
	std::wstring s;
	if (sheet)
		s = sheet->readStr(r, c);
	return QString::fromStdWString(s);
}

void xXLSReader::setViewExporter(xViewExporter* _xve)
{
	xve = _xve;
}

