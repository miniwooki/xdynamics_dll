#include "xvMeshObject.h"
#include "xvShader.h"
#include "xvAnimationController.h"
#include "xdynamics_algebra/xUtilityFunctions.h"

xvMeshObject::xvMeshObject()
: xvObject()
, vertexList(NULL)
, normalList(NULL)
, colors(NULL)
, texture(NULL)
, ntriangle(0)
, m_vertex_vbo(0)
, m_normal_vbo(0)
{
}

xvMeshObject::xvMeshObject(QString& _name)
: xvObject(V_POLYGON, _name)
, vertexList(NULL)
, normalList(NULL)
, colors(NULL)
, texture(NULL)
, ntriangle(0)
, m_vertex_vbo(0)
, m_normal_vbo(0)
{

}

xvMeshObject::~xvMeshObject()
{
	if (vertexList) delete[] vertexList; vertexList = NULL;
	// 	if (indexList) delete[] indexList; indexList = NULL;
	// 	if (vertice) delete[] vertice; vertice = NULL;
	// 	if (indice) delete[] indice; indice = NULL;
	if (normalList) delete[] normalList; normalList = NULL;
	if (colors) delete[] colors; colors = NULL;
	if (texture) delete[] texture; texture = NULL;

	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_normal_vbo){
		glDeleteBuffers(1, &m_normal_vbo);
		m_normal_vbo = 0;
	}
}

void xvMeshObject::defineMeshObject(unsigned int nt, double* v, double* n)
{
	ntriangle = nt;
	vertexList = new float[nt * 9];
	normalList = new float[nt * 9];
	for (unsigned int i = 0; i < nt; i++)
	{
		unsigned int sid = i * 9;
		for (unsigned int j = 0; j < 9; j++)
		{
			vertexList[sid + j] = static_cast<float>(v[sid + j]);
			normalList[sid + j] = static_cast<float>(n[sid + j]);
		}
	}
	if (m_vertex_vbo){
		glDeleteBuffers(1, &m_vertex_vbo);
		m_vertex_vbo = 0;
	}
	if (m_normal_vbo){
		glDeleteBuffers(1, &m_normal_vbo);
		m_normal_vbo = 0;
	}

	if (!m_vertex_vbo)
	{
		glGenBuffers(1, &m_vertex_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vertex_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ntriangle * 9, (float*)vertexList, GL_DYNAMIC_DRAW);
	}
	if (!m_normal_vbo)
	{
		glGenBuffers(1, &m_normal_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_normal_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * ntriangle * 9, (float*)normalList, GL_STATIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (!program.Program())
		program.compileProgram(polygonVertexShader, polygonFragmentShader);
	display = true;
}

//void xvMeshObject::setMeshSphere(unsigned int sz, double * d)
//{
//	if (sz != ntriangle)
//	{
//
//	}
//	r_sphere = new float[ntriangle * 3];
//	for (unsigned int i = 0; i < ntriangle; i++)
//	{
//		r_sphere[i * 3 + 0] = static_cast<float>(d[i * 3 + 0]);
//		r_sphere[i * 3 + 1] = static_cast<float>(d[i * 3 + 1]);
//		r_sphere[i * 3 + 2] = static_cast<float>(d[i * 3 + 2]);
//	}
//}

vector4f xvMeshObject::FitSphereToTriangle(vector3f& P, vector3f& Q, vector3f& R, float ft)
{
	vector3f V = Q - P;
	vector3f W = R - P;
	vector3f N = cross(V, W);
	N = N / length(N);// .length();
	vector3f M1 = (Q + P) / 2;
	vector3f M2 = (R + P) / 2;
	vector3f D1 = cross(N, V);
	vector3f D2 = cross(N, W);
	float t;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}
	vector3f Ctri = M1 + t * D1;
	vector3f Csph = new_vector3f(0.0f, 0.0f, 0.0f);
	float fc = 0;
	float tol = 1e-4f;
	float r = length(P - Ctri);
	while (abs(fc - ft) > tol)
	{
		float d = ft * r;
		float p = d / length(N);
		Csph = Ctri - p * N;
		r = length(P - Csph);
		fc = d / r;
	}
	return new_vector4f(Csph.x, Csph.y, Csph.z, r);
}

QString xvMeshObject::GenerateFitSphereFile(float ft)
{
	QString path = getenv("USERPROFILE") + QString("/Documents/xdynamics/") + Name() + ".txt";
	//		unsigned int a, b, c;
	vector3f *vertice = (vector3f*)vertexList;
	QFile qf(path);
	qf.open(QIODevice::WriteOnly);
	QTextStream qts(&qf);
	qts << name << endl;
	qts << ntriangle << endl;
	for (unsigned int i = 0; i < ntriangle; i++)
	{
		vector3f P = vertice[i * 3 + 0];
		//hpi[i].indice.x = vi++;
		vector3f Q = vertice[i * 3 + 1];
		//hpi[i].indice.y = vi++;
		vector3f R = vertice[i * 3 + 2];
		//hpi[i].indice.z = vi++;
		//vector3d ctri = xUtilityFunctions::CenterOfTriangle(hpi[i].P, hpi[i].Q, hpi[i].R);
		vector4f csph = FitSphereToTriangle(P, Q, R, ft);
		qts << csph.x << " " << csph.y << " " << csph.z << " " << csph.w << endl;
	}
	qf.close();
	return path;
}

void xvMeshObject::_drawPolygons()
{
	GLfloat ucolor[4] = { clr.redF(), clr.greenF(), clr.blueF(), clr.alphaF() };
	int loc_color = glGetUniformLocation(program.Program(), "ucolor");
	glUniform4fv(loc_color, 1, ucolor);
	if (m_vertex_vbo)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vertex_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*ntriangle * 9, vertexList);
		if (m_normal_vbo)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_normal_vbo);
			glNormalPointer(GL_FLOAT, 0, 0);
			glEnableClientState(GL_NORMAL_ARRAY);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*ntriangle * 9, normalList);
		}
		glDrawArrays(GL_TRIANGLES, 0, ntriangle * 3);
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		if (m_vertex_vbo)
			glDisableClientState(GL_VERTEX_ARRAY);
		if (m_normal_vbo)
			glDisableClientState(GL_NORMAL_ARRAY);
	}
}

void xvMeshObject::draw(GLenum eMode)
{
	if (display)
	{
		if (eMode == GL_SELECT)
		{
			glLoadName((GLuint)ID());
		}
		glPushMatrix();
		unsigned int idx = xvAnimationController::getFrame();
		bool isplaymode = (xvAnimationController::Play() || xvAnimationController::getFrame()) && xvObject::pmrs;
		if (isplaymode && xvObject::pmrs)
		{
			double t = 180 / M_PI;
			unsigned int idx = xvAnimationController::getFrame();
			xPointMass::pointmass_result pmr = xvObject::pmrs[idx];
			glTranslated(pmr.pos.x, pmr.pos.y, pmr.pos.z);
			vector3d euler = EulerParameterToEulerAngle(pmr.ep);
			glRotated(t*euler.x, 0, 0, 1);
			glRotated(t*euler.y, 1, 0, 0);
			glRotated(t*euler.z, 0, 0, 1);
			//qDebug() << euler.x << " " << euler.y << " " << euler.z;
		}
		else
		{
			glTranslated(pos.x, pos.y, pos.z);
			glRotated(ang.x, 0, 0, 1);
			glRotated(ang.y, 1, 0, 0);
			glRotated(ang.z, 0, 0, 1);
		}

	//	glColor3f(1.0f, 0.0f, 0.0f);
		glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
		glUseProgram(program.Program());
		_drawPolygons();
		glUseProgram(0);
		glPopMatrix();
	}

}

// void vpolygon::setResultData(unsigned int n)
//{
//
//}
//
//void vpolygon::insertResultData(unsigned int i, VEC3D& p, EPD& r)
//{
//
//}