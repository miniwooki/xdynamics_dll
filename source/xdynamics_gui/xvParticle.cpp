#include "xvParticle.h"
#include "xvShader.h"
#include "../xTypes.h"
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
#include <QDebug>
//#include "vcube.h"

xvParticle::xvParticle()
	: np(0)
	, isSetColor(false)
	, pos(NULL)
	, color(NULL)
	, buffers(NULL)
	, vbuffers(NULL)
	, color_buffers(NULL)
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
	if (buffers) delete[] buffers; buffers = NULL;
	if (vbuffers) delete[] vbuffers; vbuffers = NULL;
	if (color_buffers) delete[] color_buffers; color_buffers = NULL;
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
	float* pbuf = NULL;
	float* cbuf = NULL;
	if (xvAnimationController::Play())
	{
		unsigned int idx = xvAnimationController::getFrame();
		pbuf = buffers + idx * np * 4;
		cbuf = color_buffers + idx * np * 4;// model::rs->getPartColor(idx);
	}
	else
	{
		unsigned int idx = xvAnimationController::getFrame();
		if (idx)
		{
			pbuf = buffers + idx * np * 4;
			cbuf = color_buffers + idx * np * 4;
			//buffer = model::rs->getPartPosition(idx);
			//color_buffer = model::rs->getPartColor(idx);
		}
		else
		{
			pbuf = pos;
			cbuf = color;
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

	_drawPoints(pbuf, cbuf);

	glUseProgram(0);
	glDisable(GL_POINT_SPRITE_ARB);
	glEnable(GL_LIGHTING);
}

void xvParticle::_drawPoints(float* pos_buffer, float* color_buffer)
{
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (m_posVBO)
	{
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_posVBO);
		glVertexPointer(4, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * np * 4, pos_buffer);
		if (m_colorVBO)
		{
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
			glColorPointer(4, GL_FLOAT, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat) * np * 4, color_buffer);
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
	unsigned int _np = 0;
	qf.read((char*)&_np, sizeof(unsigned int));
	double* d_pos = NULL;
	while (!qf.atEnd())
	{
		qf.read((char*)&ns, sizeof(int));
		char* _name = new char[255];
		memset(_name, 0, sizeof(char) * 255);
		qf.read((char*)_name, sizeof(char) * ns);
		name.sprintf("%s", _name);
		pgds[name].name = name;
		qf.read((char*)&(pgds[name].mat), sizeof(int));
		qf.read((char*)&(pgds[name].sid), sizeof(unsigned int));
		qf.read((char*)&(pgds[name].np), sizeof(unsigned int));
		qf.read((char*)&(pgds[name].min_rad), sizeof(double));
		qf.read((char*)&(pgds[name].max_rad), sizeof(double));
		resizePositionMemory(np, np + pgds[name].np);
		d_pos = new double[pgds[name].np * 4];
		qf.read((char*)d_pos, sizeof(double) * pgds[name].np * 4);
		for (unsigned int i = 0; i < pgds[name].np; i++)
		{
			pos[(np + i) * 4 + 0] = (float)d_pos[i * 4 + 0];
			pos[(np + i) * 4 + 1] = (float)d_pos[i * 4 + 1];
			pos[(np + i) * 4 + 2] = (float)d_pos[i * 4 + 2];
			pos[(np + i) * 4 + 3] = (float)d_pos[i * 4 + 3];
		}
		np += pgds[name].np;
		delete[] d_pos;
		delete[] _name;
	}
	color = new float[np * 4];
	return _define();
}

void xvParticle::setParticlePosition(double* p, unsigned int n)
{
// 	if (pos && p)
// 		memcpy(pos, p, sizeof(double) * n * 4);
}

bool xvParticle::UploadParticleFromFile(unsigned int i, QString path)
{
	//qDebug() << i;
	double ct = 0.0;
	unsigned int inp = 0;
	unsigned int sid = 0;
	unsigned int vid = 0;
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	qf.read((char*)&ct, sizeof(double));
	qf.read((char*)&inp, sizeof(unsigned int));
	if (np != inp)
	{
		return false;
	}
	sid = inp * i * 4;
	vid = inp * i * 3;
	double *_pos = new double[inp * 4];
	double *_vel = new double[inp * 3];
	qf.read((char*)_pos, sizeof(double) * inp * 4);
	qf.read((char*)_vel, sizeof(double) * inp * 3);
	qf.close();
	xvAnimationController::addTime(i, ct);
	for (unsigned int j = 0; j < inp; j++)
	{
		unsigned int s = j * 4;
		unsigned int v = j * 3;
		buffers[s + sid + 0] = static_cast<float>(_pos[s + 0]);
		buffers[s + sid + 1] = static_cast<float>(_pos[s + 1]);
		buffers[s + sid + 2] = static_cast<float>(_pos[s + 2]);
		buffers[s + sid + 3] = static_cast<float>(_pos[s + 3]);

		vbuffers[v + vid + 0] = static_cast<float>(_vel[v + 0]);
		vbuffers[v + vid + 1] = static_cast<float>(_vel[v + 1]);
		vbuffers[v + vid + 2] = static_cast<float>(_vel[v + 2]);

		color_buffers[s + sid + 0] = 0.0f;
		color_buffers[s + sid + 1] = 0.0f;
		color_buffers[s + sid + 2] = 1.0f;
		color_buffers[s + sid + 3] = 1.0f;
	}
	delete[] _pos;
	delete[] _vel;
	return true;
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
	//buffer = pos;
	//color_buffer = color;
	if (!m_posVBO)
		m_posVBO = xvGlew::createVBO<float>(memSize, pos);
	if (!m_colorVBO)
		m_colorVBO = xvGlew::createVBO<float>(memSize, color);

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

void xvParticle::setBufferMemories(unsigned int sz)
{
	if (buffers)
		delete[] buffers;
	buffers = new float[sz * np * 4];
	if (vbuffers)
		delete[] vbuffers;
	vbuffers = new float[sz * np * 3];
	if (color_buffers)
		delete[] color_buffers;
	color_buffers = new float[sz * np * 4];
	
}

QString xvParticle::NameOfGroupData(QString& n)
{
	return pgds[n].name;
}

QString xvParticle::MaterialOfGroupData(QString& n)
{
	return NameOfMaterial(pgds[n].mat);
}

unsigned int xvParticle::NumParticlesOfGroupData(QString& n)
{
	return pgds[n].np;
}

unsigned int xvParticle::NumParticles()
{
	return np;
}

double xvParticle::MinRadiusOfGroupData(QString& n)
{
	return pgds[n].min_rad;
}

double xvParticle::MaxnRadiusOfGroupData(QString& n)
{
	return pgds[n].max_rad;
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