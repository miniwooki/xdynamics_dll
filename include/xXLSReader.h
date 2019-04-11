#ifndef XXLSREADER_H
#define XXLSREADER_H

#include "xdynamics_decl.h"
#include "libxl.h"
#include <QtCore/QString>

using namespace libxl;

class xMultiBodyModel;
class xDiscreteElementMethodModel;
class xParticleManager;
class xObjectManager;
class xContactManager;
class xViewExporter;

class XDYNAMICS_API xXLSReader
{
public:
	xXLSReader();
	~xXLSReader();

	bool Load(const wchar_t* n);
	void ReadMass(xMultiBodyModel* xmbd, vector2i rc);
	void ReadJoint(xMultiBodyModel* xmbd, vector2i rc);
	void ReadForce(xMultiBodyModel* xmbd, vector2i rc);
	void ReadParticle(xParticleManager* xparticle, vector2i rc);
	void ReadContact(xContactManager* xcm, vector2i rc);
	void ReadShapeObject(xObjectManager* xom, vector2i rc);
	void ReadIntegrator(vector2i rc);
	void ReadSimulationCondition(vector2i rc);
	bool IsEmptyCell(int r, int c);

	QString SetupSheet(int idx);
	QString ReadStr(int r, int c);

	void setViewExporter(xViewExporter* _xve);

private:
	bool _IsEmptyCell(int cid);
	xPointMassData ReadPointMassData(std::string& _name, int r, int& c);
	xJointData ReadJointData(std::string& _name, int r, int& c);
	xPlaneObjectData ReadPlaneObjectData(std::string& _name, int mat, int r, int& c);
	xCubeObjectData ReadCubeObjectData(std::string& _name, int mat, int r, int& c);
	xCubeParticleData ReadCubeParticleData(std::string& _name, int r, int& c);
	xListParticleData ReadListParticleData(std::string& _name, int r, int& c);
	xContactParameterData ReadContactData(std::string& _name, int r, int& c);
	xTSDAData ReadTSDAData(std::string& _name, int r, int& c);

	Book* book;
	Sheet* sheet;
	xViewExporter* xve;
};

#endif