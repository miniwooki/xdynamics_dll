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

int xMeshObject::DefineShapeFromFile(vector3d& loc, std::string f)
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
	com = com / ntriangle;
	double J[6] = { 0, };
	vector3d mov = loc/* - com*/;
	for (unsigned int i = 0; i < ntri; i++)
	{
// 		int s = i * 9;
// 		vertexList[s + 0] += mov.x;
// 		vertexList[s + 1] += mov.y;
// 		vertexList[s + 2] += mov.z;
// 		vertexList[s + 3] += mov.x;
// 		vertexList[s + 4] += mov.y;
// 		vertexList[s + 5] += mov.z;
// 		vertexList[s + 6] += mov.x;
// 		vertexList[s + 7] += mov.y;
// 		vertexList[s + 8] += mov.z;
		vector3d cm = spos[i] - loc;
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
	xPointMass::pos = loc;
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

QList<xMeshObject::triangle_info> xMeshObject::_splitTriangle(triangle_info& ti, double to)
{
	QList<triangle_info> added_tri;
	QList<triangle_info> temp_tri;
	//ati.push_back(ti);
	bool isAllDone = false;
	while (!isAllDone)
	{
		isAllDone = true;
		QList<triangle_info> ati;
		if (temp_tri.size())
		{
			ati = temp_tri;
			temp_tri.clear();
		}
		else
			ati.push_back(ti);
		foreach(triangle_info t, ati)
		{
			if (t.rad > to)
			{
				isAllDone = false;
				int tid = 0;
				vector3d midp;
				double s_pq = length(t.q - t.p);
				double s_qr = length(t.r - t.q);
				double s_pr = length(t.r - t.p);
				if (s_pq > s_qr)
				{
					if (s_pq > s_pr)
					{
						midp = 0.5 * (t.q + t.p);
						tid = 3;
					}
					else
					{
						midp = 0.5 * (t.r + t.p);
						tid = 2;
					}
				}
				else
				{
					if (s_qr > s_pr)
					{
						midp = 0.5 * (t.r + t.q);
						tid = 1;
					}
					else
					{
						midp = 0.5 * (t.r + t.p);
						tid = 2;
					}
				}
				vector3d aspos;//double aspos = 0.0;
				double arad = 0.0;
				vector3d an;
				vector3d p = t.p;
				vector3d q = t.q;
				vector3d r = t.r;
				if (tid == 1)
				{
					aspos = xUtilityFunctions::CenterOfTriangle(p, q, midp);
					an = cross(q - p, midp - p);
					an = an / length(an);
					arad = length(p - aspos);
					triangle_info ati0 = { arad, p, q, midp, an };
					aspos = xUtilityFunctions::CenterOfTriangle(p, midp, r);
					an = cross(midp - p, r - p);
					an = an / length(an);
					arad = length(p - aspos);
					triangle_info ati1 = { arad, p, midp, r, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
				else if (tid == 2)
				{
					aspos = xUtilityFunctions::CenterOfTriangle(q, r, midp);
					an = cross(r - q, midp - q);
					an = an / length(an);
					arad = length(q - aspos);
					triangle_info ati0 = { arad, q, r, midp, an };
					aspos = xUtilityFunctions::CenterOfTriangle(q, midp, p);
					an = cross(midp - q, p - q);
					an = an / length(an);
					arad = length(q - aspos);
					triangle_info ati1 = { arad, q, midp, p, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
				else if (tid == 3)
				{
					aspos = xUtilityFunctions::CenterOfTriangle(r, p, midp);
					an = cross(p - r, midp - r);
					an = an / length(an);
					arad = length(r - aspos);
					triangle_info ati0 = { arad, r, p, midp, an };
					aspos = xUtilityFunctions::CenterOfTriangle(r, midp, q);
					an = cross(midp - r,q - r);
					an = an / length(an);
					arad = length(r - aspos);
					triangle_info ati1 = { arad, r, midp, q, an };
					temp_tri.push_back(ati0);
					temp_tri.push_back(ati1);
				}
			}
			else
			{
				added_tri.push_back(t);
			}
		}
	}
	return added_tri;
}

QString xMeshObject::meshDataFile() const { return filePath; }
double xMeshObject::maxRadius() const { return maxRadii; }
unsigned int xMeshObject::NumTriangle() const { return ntriangle; }
double* xMeshObject::VertexList() { return vertexList; }
double* xMeshObject::NormalList(){ return normalList; }
vector3d xMeshObject::MaxPoint() const{ return max_point; }
vector3d xMeshObject::MinPoint() const{ return min_point; }

void xMeshObject::splitTriangles(double to)
{
	if (to == 0)
		return;
	vector3d p, q, r, n;

	QList<triangle_info> temp_tri;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		int  s = i * 9;
		p = new_vector3d(vertexList[s + 0], vertexList[s + 1], vertexList[s + 2]);
		q = new_vector3d(vertexList[s + 3], vertexList[s + 4], vertexList[s + 5]);
		r = new_vector3d(vertexList[s + 6], vertexList[s + 7], vertexList[s + 8]);
		n = new_vector3d(normalList[s + 0], normalList[s + 1], normalList[s + 2]);
		vector3d spos = xUtilityFunctions::CenterOfTriangle(p, q, r);
		double rad = length(spos - p);
		triangle_info tinfo = { rad, p, q, r, n };
		if (rad > to)
		{
			QList<triangle_info> added_tri = _splitTriangle(tinfo, to);
			foreach(triangle_info t, added_tri)
			{
				temp_tri.push_back(t);
			}
		}
		else
		{
			temp_tri.push_back(tinfo);
		}
	}
	//delete[] vertice;
	delete[] vertexList;
	delete[] normalList;
	ntriangle = temp_tri.size();
	//vertice = new double[ntriangle * 9];
	normalList = new double[ntriangle * 9];
	vertexList = new double[ntriangle * 9];
	int cnt = 0;
	foreach(triangle_info t, temp_tri)
	{
		int s = cnt * 9;
		vertexList[s + 0] = t.p.x;
		vertexList[s + 1] = t.p.y;
		vertexList[s + 2] = t.p.z;
		vertexList[s + 3] = t.q.x;
		vertexList[s + 4] = t.q.y;
		vertexList[s + 5] = t.q.z;
		vertexList[s + 6] = t.r.x;
		vertexList[s + 7] = t.r.y;
		vertexList[s + 8] = t.r.z;

		normalList[s + 0] = t.n.x;
		normalList[s + 1] = t.n.y;
		normalList[s + 2] = t.n.z;
		normalList[s + 3] = t.n.x;
		normalList[s + 4] = t.n.y;
		normalList[s + 5] = t.n.z;
		normalList[s + 6] = t.n.x;
		normalList[s + 7] = t.n.y;
		normalList[s + 8] = t.n.z;
		cnt++;
	}
}

void xMeshObject::translation(vector3d p)
{
	vector3d mov = p - xPointMass::pos;
	///*xPointMass*/::translation(p);
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		int s = i * 9;
		vertexList[s + 0] += mov.x;
		vertexList[s + 1] += mov.y;
		vertexList[s + 2] += mov.z;
		vertexList[s + 3] += mov.x;
		vertexList[s + 4] += mov.y;
		vertexList[s + 5] += mov.z;
		vertexList[s + 6] += mov.x;
		vertexList[s + 7] += mov.y;
		vertexList[s + 8] += mov.z;
		//vector3d cm = spos[i] - pos;
// 		J[0] += cm.y * cm.y + cm.z * cm.z;
// 		J[1] += cm.x * cm.x + cm.z * cm.z;
// 		J[2] += cm.x * cm.x + cm.y * cm.y;
// 		J[3] -= cm.x * cm.y;
// 		J[4] -= cm.x * cm.z;
// 		J[5] -= cm.y * cm.z;
	}
}

std::string xMeshObject::exportMeshData(std::string path)
{
	std::fstream fs;
	//std::cout << path << std::endl;
	fs.open(path, std::ios::out | std::ios::binary);
	unsigned int ns = static_cast<unsigned int>(name.size());
	fs.write((char*)&ns, sizeof(unsigned int));
	fs.write((char*)xUtilityFunctions::xstring(name).c_str(), sizeof(char)*ns);
	double *_vertex = this->VertexList();
	double *_normal = this->NormalList();
	fs.write((char*)&material, sizeof(int));
	fs.write((char*)&pos, sizeof(double) * 3);
	unsigned int nt = this->NumTriangle();
	fs.write((char*)&nt, sizeof(unsigned int));
	fs.write((char*)_vertex, sizeof(double) * this->NumTriangle() * 9);
	fs.write((char*)_normal, sizeof(double) * this->NumTriangle() * 9);
	fs.close();	
	return path;
}

unsigned int xMeshObject::create_sph_particles(double ps, unsigned int nlayers, vector3d* p, xMaterialType* t)
{
	return 0;
}

QVector<xCorner> xMeshObject::get_sph_boundary_corners()
{
	return QVector<xCorner>();
}

//unsigned int* xMeshObject::IndexList() { return indexList; }