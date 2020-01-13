#ifndef XXLSREADER_H
#define XXLSREADER_H

#include "xdynamics_decl.h"
//#include "libxl.h"
#include "xstring.h"

//#include <shellapi.h>
//#include "xString."
//#include <QtCore/QString>

//using namespace libxl;.using namespace ExcelFormat;
//using namespace ExcelFormat;
namespace ExcelFormat
{
	struct XLSFormatManager;
	
}
namespace YExcel
{
	class BasicExcel;
	class BasicExcelWorksheet;
}
////namespace VExcel
////{
////	
////}


class xMultiBodyModel;
class xDiscreteElementMethodModel;
class xSmoothedParticleHydrodynamicsModel;
class xParticleManager;
class xObjectManager;
class xContactManager;
class xViewExporter;

class XDYNAMICS_API xXLSReader
{
public:
	xXLSReader();
	~xXLSReader();

	bool Load(const char* n);
	void Release();
	void ReadMass(xMultiBodyModel* xmbd, vector2i rc);
	void ReadJoint(xMultiBodyModel* xmbd, vector2i rc);
	void ReadForce(xMultiBodyModel* xmbd, xDiscreteElementMethodModel* xdem, vector2i rc);
	void ReadKernel(xSmoothedParticleHydrodynamicsModel* xsph, vector2i rc);
	void ReadDEMParticle(xDiscreteElementMethodModel* xdem, xObjectManager* xom, vector2i rc);
	void ReadSPHParticle(xSmoothedParticleHydrodynamicsModel* xsph, vector2i rc);
	void ReadContact(xContactManager* xcm, vector2i rc);
	void ReadShapeObject(xObjectManager* xom, vector2i rc);
	void ReadIntegrator(vector2i rc);
	void ReadSimulationCondition(vector2i rc);
	void ReadInputGravity(vector2i rc);
	bool IsEmptyCell(int r, int c);

	xstring SetupSheet(int idx);
	std::string ReadStr(int r, int c);
	double ReadNum(int r, int c);

	void setViewExporter(xViewExporter* _xve);

private:
	bool _IsEmptyCell(int cid);
	xPointMassData ReadPointMassData(std::string& _name, int r, int& c, bool v = true);
	xJointData ReadJointData(std::string& _name, int r, int& c);
	xLineObjectData ReadLineObjectData(std::string& _name, int mat, int r, int& c);
	xPlaneObjectData ReadPlaneObjectData(std::string& _name, int mat, int r, int& c);
	xCubeObjectData ReadCubeObjectData(std::string& _name, int mat, int r, int& c);
	xCubeParticleData ReadCubeParticleData(std::string& _name, int r, int& c);
	xPlaneParticleData ReadPlaneParticleData(std::string& _name, int r, int& c);
	xLineParticleData ReadLineParticleData(std::string& _name, int r, int& c);
	xListParticleData ReadListParticleData(std::string& _name, int r, int& c);
	xCircleParticleData ReadCircleParticleData(std::string& _name, int r, int& c);
	xContactParameterData ReadContactData(std::string& _name, int r, int& c);
	xCylinderObjectData ReadCylinderObjectData(std::string& _name, int mat, int r, int& c);
	xTSDAData ReadTSDAData(std::string& _name, int r, int& c);
	xRSDAData ReadRSDAData(std::string& _name, int r, int& c);
	xRotationalAxialForceData ReadxRotationalAxialForceData(std::string& _name, int r, int& c);
	//xSPHPlaneObjectData ReadSPHPlaneParticleData(std::string& _name, int r, int& c);
	//B xls;
	ExcelFormat::XLSFormatManager* fmgr;
	YExcel::BasicExcel* xls;
	YExcel::BasicExcelWorksheet* sheet;
	//Book* book;
	//Sheet* sheet;
	xViewExporter* xve;
	xstring connect_file;
};

#endif