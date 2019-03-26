#include "xvParticle.h"
#include "xvShader.h"
//#include "colors.h"
//#include "msgBox.h"
//#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <QStringList>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QGridLayout>
#include <QComboBox>
#include <QTabWidget>
#include <QDialogButtonBox>

//#include "vcube.h"

xvParticle::xvParticle()
	: np(0)
	, isSetColor(false)
	, pos(NULL)
	, color(NULL)
	, buffer(NULL)
	, color_buffer(NULL)
	, pscale(0)
	, isDefine(false)
{
	m_posVBO = 0;
	m_colorVBO = 0;
	m_program = 0;
}

xvParticle::~xvParticle()
{
	if (pos) delete[] pos; pos = NULL;
	if (color) delete[] color; color = NULL;
	if (m_posVBO){
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO){
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	program.deleteProgram();
}

void xvParticle::draw(GLenum eModem, int wHeight, int protype, double z)
{
	if (!isDefine)
		return;
	if (xvAnimationController::Play())
	{
		unsigned int idx = xvAnimationController::getFrame();
		//buffer = model::rs->getPartPosition(idx);
		//color_buffer = model::rs->getPartColor(idx);
	}
	else
	{
		unsigned int idx = xvAnimationController::getFrame();
		if (idx)
		{
			//buffer = model::rs->getPartPosition(idx);
			//color_buffer = model::rs->getPartColor(idx);
		}
		else
		{
			buffer = pos;
			color_buffer = color;
		}
	}

	glDisable(GL_LIGHTING);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(program.Program());
	float projFactor = 0.f;
	if (!protype){
		// 		float fTmp1[16] = { 0.f, };
		// 		glGetFloatv(GL_MODELVIEW_MATRIX, fTmp1);
		//  		pscale = wHeight * 1.34;
		//  		glUniform1f(glGetUniformLocation(m_program, "orthoDist"), abs(fTmp1[14]));
		pscale = (wHeight) / (0.6 * z);// / tanf(90 * 0.5*M_PI / 180.0);
		projFactor = 1.f;
	}
	else{
		pscale = (wHeight) / tanf(60 * 0.48*M_PI / 180.0);
	}
	//float ptype = 100.f;
	//glUniform1f(glGetUniformLocation(m_program, "isOrgho"), projFactor);
	//glUniform1f(glGetUniformLocation(m_program, "pointScale"), (wHeight) / tanf(60 * 0.5*M_PI / 180.0));
	glUniform1f(glGetUniformLocation(program.Program(), "projFactor"), projFactor);
	glUniform1f(glGetUniformLocation(program.Program(), "pointScale"), pscale);
	//glUniform1f(glGetUniformLocation(m_program, "orthoDist"), abs(fTmp1[14]));

	_drawPoints();

	glUseProgram(0);
	glDisable(GL_POINT_SPRITE_ARB);
	glEnable(GL_LIGHTING);
}

void xvParticle::_drawPoints()
{
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (m_posVBO)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_posVBO);
		glVertexPointer(4, GL_DOUBLE, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*np * 4, buffer);
		if (m_colorVBO)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
			glColorPointer(4, GL_DOUBLE, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*np * 4, color_buffer);
		}

		glDrawArrays(GL_POINTS, 0, np);

		glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
}

// bool xvParticle::define()
// {
// // 	np = 0;// ps->numParticle();
// // 	if (!pos) pos = new float[np * 4];
// // 	if (!vel) vel = new float[np * 3];
// // 	if (!force) force = new float[np * 3];
// // 	if (!color) color = new float[np * 4];
// // 
// // 	//ps->setPosition(pos);
// // 	for (unsigned int i = 0; i < np; i++){
// // 		color[i * 4 + 0] = 0.0f;
// // 		color[i * 4 + 1] = 0.0f;
// // 		color[i * 4 + 2] = 1.0f;
// // 		color[i * 4 + 3] = 1.0f;
// // 	}
// // 	if (!np){
// // 		msgBox("Particle generation is failed.", QMessageBox::Critical);
// // 		return false;
// // 	}
// // 	// 	if (!isglewinit)
// // 	// 		glewInit();
// // 
// // 	if (m_posVBO){
// // 		glDeleteBuffers(1, &m_posVBO);
// // 		m_posVBO = 0;
// // 	}
// // 	if (m_colorVBO){
// // 		glDeleteBuffers(1, &m_colorVBO);
// // 		m_colorVBO = 0;
// // 	}
// // 	if (program.Program())
// // 		program.deleteProgram();
// // 	unsigned int memSize = sizeof(double) * 4 * np;
// // 	buffer = pos;
// // 	color_buffer = color;
// // 	if (!m_posVBO)
// // 		m_posVBO = vglew::createVBO<double>(memSize, buffer);
// // 	if (!m_colorVBO)
// // 		m_colorVBO = vglew::createVBO<double>(memSize, color_buffer);
// // 
// // 	if (!program.Program())
// // 		program.compileProgram(vertexShader, spherePixelShader);
// 
// 	return true;
// }


bool xvParticle::defineFromViewFile(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	int ns = 0;
	QString name;
	while (!qf.atEnd())
	{
		qf.read((char*)&ns, sizeof(int));
		char* _name = new char[255];
		memset(_name, 0, sizeof(char) * 255);
		qf.read((char*)_name, sizeof(char) * ns);
		name.sprintf("%s", _name);
		qf.read((char*)&(pgds[name].mat), sizeof(int));
		qf.read((char*)&(pgds[name].sid), sizeof(unsigned int));
		qf.read((char*)&(pgds[name].np), sizeof(unsigned int));
		qf.read((char*)&(pgds[name].min_rad), sizeof(double));
		qf.read((char*)&(pgds[name].max_rad), sizeof(double));
		resizePositionMemory(np, np + pgds[name].np);
		np += pgds[name].np;
	}
	color = new float[np * 4];
}

void xvParticle::setParticlePosition(double* p, unsigned int n)
{
// 	if (pos && p)
// 		memcpy(pos, p, sizeof(double) * n * 4);
}

void xvParticle::resizePositionMemory(unsigned int n0, unsigned int n1)
{
	if (!n0)
	{
		pos = new float[n1 * 4];
	}
	else
	{
		float* temp = new float[n0 * 4];
		memcpy(temp, pos, sizeof(float) * n0 * 4);
		delete[] pos;
		pos = new float[n1 * 4];
		memset(pos, 0, sizeof(float) * n1 * 4);
		memcpy(pos, temp, sizeof(float) * n0 * 4);
		delete[] temp;
	}
	
// 	double* tv4 = new double[np * 4];
// 	double* tv3 = new double[np * 3];
// 	unsigned int new_np = n;// ps->numParticle();
// 	memcpy(tv4, pos, sizeof(double) * np * 4);
// 	delete[] pos;
// 	pos = new double[new_np * 4]; memcpy(pos, p, sizeof(double) * new_np * 4);
// 	memcpy(tv3, vel, sizeof(double) * np * 3); delete[] vel; vel = new double[new_np * 3]; memcpy(vel, tv3, sizeof(double) * np * 3);
// 	memcpy(tv3, force, sizeof(double) * np * 3); delete[] force; force = new double[new_np * 3]; memcpy(force, tv3, sizeof(double) * np * 3);
// 	memcpy(tv4, color, sizeof(double) * np * 4); delete[] color; color = new double[new_np * 4]; memcpy(color, tv4, sizeof(double) * np * 4);
// 	delete[] tv4;
// 	delete[] tv3;
// 	np = new_np;
	//define(p, n);
}

bool xvParticle::_define()
{
	for (unsigned int i = 0; i < np; i++)
	{
		color[i * 4 + 0] = 0.0f;
		color[i * 4 + 1] = 0.0f;
		color[i * 4 + 2] = 1.0f;
		color[i * 4 + 3] = 1.0f;
	}
// 
// 	if (!np){
// 		msgBox("Particle generation is failed.", QMessageBox::Critical);
// 		return false;
// 	}
// 	// 	if (!isglewinit)
// 	// 		glewInit();

	if (m_posVBO)
	{
		glDeleteBuffers(1, &m_posVBO);
		m_posVBO = 0;
	}
	if (m_colorVBO)
	{
		glDeleteBuffers(1, &m_colorVBO);
		m_colorVBO = 0;
	}
	if (program.Program())
		program.deleteProgram();
	unsigned int memSize = sizeof(float) * 4 * np;
	buffer = pos;
	color_buffer = color;
	if (!m_posVBO)
		m_posVBO = xvGlew::createVBO<float>(memSize, buffer);
	if (!m_colorVBO)
		m_colorVBO = xvGlew::createVBO<float>(memSize, color_buffer);

	if (!program.Program())
		program.compileProgram(vertexShader, spherePixelShader);
	isDefine = true;
	return true;
}

// unsigned int xvParticle::createVBO(unsigned int size, float *bufferData)
// {
// 	GLuint vbo;
// 	glGenBuffers(1, &vbo);
// 	glBindBuffer(GL_ARRAY_BUFFER, vbo);
// 	glBufferData(GL_ARRAY_BUFFER, size, bufferData, GL_DYNAMIC_DRAW);
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
// 	
// 	return vbo;
// }
// 
// unsigned int xvParticle::_compileProgram(const char *vsource, const char *fsource)
// {
// 	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
// 	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
// 
// 	glShaderSource(vertexShader, 1, &vsource, 0);
// 	glShaderSource(fragmentShader, 1, &fsource, 0);
// 
// 	glCompileShader(vertexShader);
// 	glCompileShader(fragmentShader);
// 
// 	GLuint program = glCreateProgram();
// 
// 	glAttachShader(program, vertexShader);
// 	glAttachShader(program, fragmentShader);
// 
// 	glLinkProgram(program);
// 
// 	// check if program linked
// 	GLint success = 0;
// 	glGetProgramiv(program, GL_LINK_STATUS, &success);
// 
// 	if (!success) {
// 		char temp[256];
// 		glGetProgramInfoLog(program, 256, 0, temp);
// 		printf("Failed to link program:\n%s\n", temp);
// 		glDeleteProgram(program);
// 		program = 0;
// 	}
// 
// 	return program;
// }

void xvParticle::openResultFromFile(unsigned int idx)
{
	// 	QFile pf(rList.at(vcontroller::getFrame()));
	// 	pf.open(QIODevice::ReadOnly);
	// 	double time = 0.f;
	// 	unsigned int _np = 0;
	// 	pf.read((char*)&_np, sizeof(unsigned int));
	// 	pf.read((char*)&time, sizeof(double));
	// 	pf.read((char*)pos, sizeof(double) * 4 * np);
	// 	pf.read((char*)vel, sizeof(double) * 3 * np);
	// 	pf.read((char*)force, sizeof(double) * 3 * np);
	// 
	// // 	double grad = MPForce * 0.1f;
	// // 	double t = 0.f;
	// // 	for (unsigned int i = 0; i < np; i++){
	// // 		double m = VEC3F(force[i * 3 + 0], force[i * 3 + 1], force[i * 3 + 2]).length();
	// // 		t = (m - 0) / grad;
	// // 		if (t > 7)
	// // 			m = m;
	// // 		ucolors::colorRamp(t, &(color[i * 4]));
	// // 		color[i * 4 + 3] = 1.0f;
	// // 		//break;
	// // 		//}
	// // 	}
	// // 	color[203878 * 4 + 0] = 1.0f;
	// // 	color[203878 * 4 + 1] = 1.0f;
	// // 	color[203878 * 4 + 2] = 1.0f;
	// 	pf.close();
}

// void xvParticle::changeParticles(VEC4D_PTR _pos)
// {
// 	memcpy(pos, _pos, sizeof(double) * 4 * np);
// }

// void xvParticle::calcMaxForce()
// {
// 	MPForce = 0.f;
// 	double *v4 = new double[np * 4];
// 	double *v3 = new double[np * 3];
// 	for (unsigned int i = 0; i < rList.size(); i++){
// 		QFile pf(rList.at(i));
// 		pf.open(QIODevice::ReadOnly);
// 		double time = 0.f;
// 		unsigned int _np = 0;
// 
// 		pf.read((char*)&_np, sizeof(unsigned int));
// 		pf.read((char*)&time, sizeof(double));
// 		pf.read((char*)v4, sizeof(double) * 4 * np);
// 		pf.read((char*)v3, sizeof(double) * 3 * np);
// 		pf.read((char*)v3, sizeof(double) * 3 * np);
// 		for (unsigned int i = 0; i < np; i++){
// 			double m = sqrt(v3[i * 3 + 0] * v3[i * 3 + 0] + v3[i * 3 + 1] * v3[i * 3 + 1] + v3[i * 3 + 2] * v3[i * 3 + 2]);
// 			if (MPForce < m)
// 				MPForce = m;
// 		}
// 	}
// 	delete[] v4; v4 = NULL;
// 	delete[] v3; v3 = NULL;
// }