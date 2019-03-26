#include "xdynamics_object/xMeshObject.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include <fstream>

xMeshObject::xMeshObject()
	: xPointMass()
	, vertexList(NULL)
	//, indexList(NULL)
	, ntriangle(0)
	, maxRadii(0)
	, filePath("")
	, max_point(new_vector3d(0, 0, 0))
	, min_point(new_vector3d(0, 0, 0))
{

}

xMeshObject::xMeshObject(std::string _name)
	: xPointMass(_name, MESH_SHAPE)
	, vertexList(NULL)
	, maxRadii(0)
	, ntriangle(0)
	, filePath("")
	, max_point(new_vector3d(0, 0, 0))
	, min_point(new_vector3d(0, 0, 0))
{
}

xMeshObject::xMeshObject(const xMeshObject& mesh)
	: xPointMass(*this)
	, vertexList(NULL)
	//, indexList(NULL)
	, ntriangle(mesh.NumTriangle())
	, maxRadii(mesh.maxRadius())
	, filePath(mesh.meshDataFile())
	, max_point(mesh.MaxPoint())
	, min_point(mesh.MinPoint())
{
}

xMeshObject::~xMeshObject()
{
	if (vertexList) delete[] vertexList; vertexList = NULL;
	if (normalList) delete[] normalList; normalList = NULL;
	//if (indexList) delete[] indexList; indexList = NULL;
}

bool xMeshObject::define(xImportShapeType t, vector3d& loc, int _ntriangle, double* vList, unsigned int *iList)
{
	this->setPosition(loc.x, loc.y, loc.z);
	switch (t)
	{
	//case MILKSHAPE_3D_ASCII: _fromMS3DASCII(_ntriangle, vList, iList); break;
	case STL_ASCII: _fromSTLASCII(_ntriangle, vList, loc); break;
	}
	return true;
}

int xMeshObject::DefineShapeFromFile(std::string f)
{
	std::fstream ofs;
	ofs.open(f, std::ios::in);
	std::string ch;
	ofs >> ch >> ch >> ch;
	unsigned int ntri = 0;
	while (!ofs.eof())
	{
		ofs >> ch;
		if (ch == "facet")
			ntri++;
	}
	vertexList = new double[ntri * 9];
	normalList = new double[ntri * 9];
	double x, y, z;
	double nx, ny, nz;
	ofs.close();
	ofs.open(f, std::ios::in);
	//ofs.seekg(0, ios::beg);
	ofs >> ch >> ch >> ch;
	vector3d p, q, r;// , c;
	vector3d com = new_vector3d(0.0, 0.0, 0.0);
	double _vol = 0.0;
	double min_radius = 10000.0;
	double max_radius = 0.0;
	min_point = { FLT_MAX, FLT_MAX, FLT_MAX };
	max_point = { FLT_MIN, FLT_MIN, FLT_MIN };
	vector3d *spos = new vector3d[ntri];
//	ixx = iyy = izz = ixy = ixz = iyz = 0.0;
	//unsigned int nc = 0;
	for (unsigned int i = 0; i < ntri; i++)
	{
		ofs >> ch >> ch >> nx >> ny >> nz;
		normalList[i * 9 + 0] = nx;
		normalList[i * 9 + 1] = ny;
		normalList[i * 9 + 2] = nz;
		normalList[i * 9 + 3] = nx;
		normalList[i * 9 + 4] = ny;
		normalList[i * 9 + 5] = nz;
		normalList[i * 9 + 6] = nx;
		normalList[i * 9 + 7] = ny;
		normalList[i * 9 + 8] = nz;
		ofs >> ch >> ch;
		ofs >> ch >> x >> y >> z;
		p.x = vertexList[i * 9 + 0] = 0.001 * x;
		p.y = vertexList[i * 9 + 1] = 0.001 * y;
		p.z = vertexList[i * 9 + 2] = 0.001 * z;
		vertexList[i * 9 + 0] = vertexList[i * 9 + 0];
		vertexList[i * 9 + 1] = vertexList[i * 9 + 1];
		vertexList[i * 9 + 2] = vertexList[i * 9 + 2];

		ofs >> ch >> x >> y >> z;
		q.x = vertexList[i * 9 + 3] = 0.001 * x;
		q.y = vertexList[i * 9 + 4] = 0.001 * y;
		q.z = vertexList[i * 9 + 5] = 0.001 * z;
		vertexList[i * 9 + 3] = vertexList[i * 9 + 3];
		vertexList[i * 9 + 4] = vertexList[i * 9 + 4];
		vertexList[i * 9 + 5] = vertexList[i * 9 + 5];

		ofs >> ch >> x >> y >> z;
		r.x = vertexList[i * 9 + 6] = 0.001 * x;
		r.y = vertexList[i * 9 + 7] = 0.001 * y;
		r.z = vertexList[i * 9 + 8] = 0.001 * z;
		vertexList[i * 9 + 6] = vertexList[i * 9 + 6];
		vertexList[i * 9 + 7] = vertexList[i * 9 + 7];
		vertexList[i * 9 + 8] = vertexList[i * 9 + 8];
		ofs >> ch >> ch;
		_vol += xUtilityFunctions::SignedVolumeOfTriangle(p, q, r);
		spos[i] = xUtilityFunctions::CenterOfTriangle(p, q, r);
		com += spos[i];
		double _r = length(spos[i] - p);
		if (max_radius < _r) max_radius = _r;
		if (min_radius > _r) min_radius = _r;
		min_point.x = xmin(xmin(p.x, q.x, r.x), min_point.x);
		min_point.y = xmin(xmin(p.y, q.y, r.y), min_point.y);
		min_point.z = xmin(xmin(p.z, q.z, r.z), min_point.z);
		max_point.x = xmax(xmax(p.x, q.x, r.x), max_point.x);
		max_point.y = xmax(xmax(p.y, q.y, r.y), max_point.y);
		max_point.z = xmax(xmax(p.z, q.z, r.z), max_point.z);

	}
	ntriangle = ntri;
	double J[6] = { 0, };
	for (unsigned int i = 0; i < ntri; i++)
	{
		int s = i * 9;
		vertexList[s + 0] -= pos.x;
		vertexList[s + 1] -= pos.y;
		vertexList[s + 2] -= pos.z;
		vertexList[s + 3] -= pos.x;
		vertexList[s + 4] -= pos.y;
		vertexList[s + 5] -= pos.z;
		vertexList[s + 6] -= pos.x;
		vertexList[s + 7] -= pos.y;
		vertexList[s + 8] -= pos.z;
		vector3d cm = spos[i] - pos;
		J[0] += cm.y * cm.y + cm.z * cm.z;
		J[1] += cm.x * cm.x + cm.z * cm.z;
		J[2] += cm.x * cm.x + cm.y * cm.y;
		J[3] -= cm.x * cm.y;
		J[4] -= cm.x * cm.z;
		J[5] -= cm.y * cm.z;
	}
	ofs.close();
	xPointMass::diag_inertia = new_vector3d(J[0], J[1], J[2]);
	xPointMass::syme_inertia = new_vector3d(J[3], J[4], J[5]);
	vol = _vol;
	xPointMass::mass = this->Density() * vol;
	xPointMass::pos = com / ntriangle;
	delete[] spos;
	return 0;// xDynamicsError::xdynamicsSuccess;
	//nvtriangle = ntriangle;
}

void xMeshObject::_fromSTLASCII(int _ntriangle, double* vList, vector3d& loc)
{
	ntriangle = _ntriangle;
	vertexList = vList;
	vector3d P, Q, R;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		P = new_vector3d(vertexList[i * 9 + 0], vertexList[i * 9 + 1], vertexList[i * 9 + 2]);
		Q = new_vector3d(vertexList[i * 9 + 3], vertexList[i * 9 + 4], vertexList[i * 9 + 5]);
		R = new_vector3d(vertexList[i * 9 + 6], vertexList[i * 9 + 7], vertexList[i * 9 + 8]);
		P = this->toLocal(P - pos);
		Q = this->toLocal(Q - pos);
		R = this->toLocal(R - pos);
		vertexList[i * 9 + 0] = P.x;
		vertexList[i * 9 + 1] = P.y;
		vertexList[i * 9 + 2] = P.z;
		vertexList[i * 9 + 3] = Q.x;
		vertexList[i * 9 + 4] = Q.y;
		vertexList[i * 9 + 5] = Q.z;
		vertexList[i * 9 + 6] = R.x;
		vertexList[i * 9 + 7] = R.y;
		vertexList[i * 9 + 8] = R.z;
	}
}

QString xMeshObject::meshDataFile() const { return filePath; }
double xMeshObject::maxRadius() const { return maxRadii; }
unsigned int xMeshObject::NumTriangle() const { return ntriangle; }
double* xMeshObject::VertexList() { return vertexList; }
double* xMeshObject::NormalList(){ return normalList; }
vector3d xMeshObject::MaxPoint() const{ return max_point; }
vector3d xMeshObject::MinPoint() const{ return min_point; }
//unsigned int* xMeshObject::IndexList() { return indexList; }