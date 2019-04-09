#include "xvMeshObject.h"
#include "xvShader.h"
#include "xvAnimationController.h"

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
		if (idx != 0)
		{
// 			if (model::rs->pointMassResults().find(nm) != model::rs->pointMassResults().end())
// 			{
// 				VEC3D p = model::rs->pointMassResults()[nm].at(idx).pos;
// 				//	qDebug() << p.x << " " << p.y << " " << p.z;
// 				EPD ep = model::rs->pointMassResults()[nm].at(idx).ep;
// 				animationFrame(p, ep);// p.x, p.y, p.z);
// 			}
// 			else
// 			{
// 				glTranslated(pos0.x, pos0.y, pos0.z);
// 				glRotated(ang0.x, 0, 0, 1);
// 				glRotated(ang0.y, 1, 0, 0);
// 				glRotated(ang0.z, 0, 0, 1);
// 			}
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