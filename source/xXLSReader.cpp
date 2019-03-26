#include "xXLSReader.h"
#include "xViewExporter.h"
#include "xdynamics_manager/xMultiBodyModel.h"
#include "xdynamics_manager/XDiscreteElementMethodModel.h"
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

xPlaneObjectData xXLSReader::ReadPlaneObjectData(std::string& _name, int mat, int r, int& c)
{
	xPlaneObjectData d = { 0, };
	double *ptr = &d.dx;
	std::wstring x;
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 2, ptr + 0);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 2);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 5);
	x = sheet->readStr(r, c++); uf::xsplit(x, ",", 3, ptr + 8);
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
// 	if (xve)
// 	{
// 		int t = VPARTICLE;
// 		xve->Write((char*)&t, sizeof(int));
// 		int ns = _name.size(); xve->Write((char*)&ns, sizeof(int));
// 		xve->Write((char*)_name.c_str(), sizeof(char)*_name.size());
// 		xve->Write((char*)&d, sizeof(xCubeParticleData));
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
		//std::wstring name, str;			
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y)) break;
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xPointMass* xpm = xmbd->CreatePointMass(name);
			xpm->SetDataFromStructure(xmbd->NumMass(), ReadPointMassData(name, rc.x++, rc.y));
			rc.y = 0;
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
		case xForce::TSDA: (dynamic_cast<xSpringDamperForce*>(xf))->SetupDataFromStructure(ReadTSDAData(name, rc.x, rc.y)); break;
		}
	}
}

void xXLSReader::ReadParticle(xParticleManager* xparticle, vector2i rc)
{
	if (xparticle)
	{
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y))
				break;
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xShapeType form = static_cast<xShapeType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			int material = static_cast<int>(sheet->readNum(rc.x, rc.y++));
			if (form == CUBE_SHAPE)
			{
				xCubeParticleData d = ReadCubeParticleData(name, rc.x++, rc.y);
				unsigned int np = xparticle->GetNumCubeParticles(d.dx, d.dy, d.dz, d.minr, d.maxr);
				xparticle->CreateCubeParticle(name.c_str(), (xMaterialType)material, np, d);
			}
			else if (form == NO_SHAPE_AND_LIST)
			{
				unsigned int number = static_cast<int>(sheet->readNum(rc.x, rc.y++));
				std::wstring x;
				x = sheet->readStr(rc.x, rc.y); 
				uf::xsplit(x, ",", 2, &rc.x);
				vector4d* d = new vector4d[number];
				rc.x -= 1; rc.y -= 1;
				for (unsigned int i = 0; i < number; i++)
				{
					if (IsEmptyCell(rc.x, rc.y))
						break;
					d[i].x = sheet->readNum(rc.x, rc.y++);
					d[i].y = sheet->readNum(rc.x, rc.y++);
					d[i].z = sheet->readNum(rc.x, rc.y++);
					d[i].w = sheet->readNum(rc.x++, rc.y);
				}
				xparticle->CreateParticleFromList(name.c_str(), (xMaterialType)material, number, d);
			}
		}
	}
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
		while (1)
		{
			if (IsEmptyCell(rc.x, rc.y)) break;
			std::string name = xUtilityFunctions::WideChar2String(sheet->readStr(rc.x, rc.y++));
			xShapeType form = static_cast<xShapeType>(static_cast<int>(sheet->readNum(rc.x, rc.y++)));
			int material = static_cast<int>(sheet->readNum(rc.x, rc.y++));
			if (form == xShapeType::CUBE_SHAPE)
			{
				xCubeObject* xco = xom->CreateCubeShapeObject(name, material);
				xco->SetupDataFromStructure(ReadCubeObjectData(name, material, rc.x++, rc.y));
			}
			else if (form == xShapeType::PLANE_SHAPE)
			{
				xPlaneObject* xpo = xom->CreatePlaneShapeObject(name, material);
				xpo->SetupDataFromStructure(ReadPlaneObjectData(name, material, rc.x++, rc.y));
			}
			else if (form == xShapeType::MESH_SHAPE)
			{
				xMeshObject* xmo = xom->CreateMeshShapeObject(name, material);
				std::string mf = uf::WideChar2String(sheet->readStr(rc.x, rc.y++));
				xmo->DefineShapeFromFile(mf);
				if (xve)
				{
					int t = VMESH;
					xve->Write((char*)&t, sizeof(int));
					unsigned int ns = static_cast<unsigned int>(name.size()); 
					xve->Write((char*)&ns, sizeof(unsigned int));
					xve->Write((char*)name.c_str(), sizeof(char)*ns);
					double *_vertex = xmo->VertexList();
					double *_normal = xmo->NormalList();
					xve->Write((char*)&material, sizeof(int));
					xve->Write((char*)_vertex, sizeof(double) * xmo->NumTriangle() * 9);
					xve->Write((char*)_normal, sizeof(double) * xmo->NumTriangle() * 9);
				}
			}
		}
	}
}

void xXLSReader::ReadIntegrator(vector2i rc)
{
	std::wstring sol;
	int ic = rc.y;
	if (!IsEmptyCell(rc.x, rc.y))
	{
		sol = sheet->readStr(rc.x, rc.y++);
		if (sol == L"MBD")
		{
			xSimulation::MBDSolverType type = static_cast<xSimulation::MBDSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setMBDSolverType(type);
		}
	}
	rc.y = ic;
	if (!IsEmptyCell(rc.x, rc.y))
	{
		sol = sheet->readStr(rc.x, rc.y++);
		if (sol == L"DEM")
		{
			xSimulation::DEMSolverType type = static_cast<xSimulation::DEMSolverType>(static_cast<int>(sheet->readNum(rc.x++, rc.y)));
			xSimulation::setDEMSolverType(type);
		}
	}
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

