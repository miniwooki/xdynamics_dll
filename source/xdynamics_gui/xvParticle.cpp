#include "xvParticle.h"
#include "xvShader.h"
#include "../xTypes.h"
#include "xdynamics_algebra/xAlgebraMath.h"
#include "xdynamics_object/xParticleObject.h"
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
#include "..\..\include\xdynamics_gui\xvParticle.h"
//#include "vcube.h"

xvParticle::xvParticle()
	: np(0)
	, r_np(0)
	, isSetColor(false)
	, pos(NULL)
	, color(NULL)
	, r_pos(NULL)
	, pscale(0)
	, isDefine(false)
	, min_radius(FLT_MAX)
	, max_radius(-FLT_MAX)
{
	m_posVBO = 0;
	m_colorVBO = 0;
	m_program = 0;
	pbuffers = vbuffers = cbuffers = NULL;
}

xvParticle::~xvParticle()
{
	if (pos) delete[] pos; pos = NULL;
	if (color) delete[] color; color = NULL;
	if (r_pos) delete[] r_pos; r_pos = NULL;
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
		pbuf = pbuffers + idx * np * 4;
		cbuf = cbuffers + idx * np * 4;// model::rs->getPartColor(idx);
	}
	else
	{
		unsigned int idx = xvAnimationController::getFrame();
		if (idx)
		{
			pbuf = pbuffers + idx * np * 4;
			cbuf = cbuffers + idx * np * 4;
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


void xvParticle::bind_result_buffers(float * pos_buffer, float * vel_buffer, float * color_buffer)
{
	pbuffers = pos_buffer;
	vbuffers = vel_buffer;
	cbuffers = color_buffer;
}

bool xvParticle::defineFromParticleObject(xParticleObject* pobj)
{
	resizePositionMemory(np, np + pobj->NumParticle());
	vector4d *d_pos = pobj->Position();
	for (unsigned int i = 0; i < pobj->NumParticle(); i++) {
		pos[(np + i) * 4 + 0] = (float)d_pos[i].x;
		pos[(np + i) * 4 + 1] = (float)d_pos[i].y;
		pos[(np + i) * 4 + 2] = (float)d_pos[i].z;
		double rad = pos[(np + i) * 4 + 3] = (float)d_pos[i].w;
		if (rad < min_radius)
			min_radius = rad;
		if (rad > max_radius)
			max_radius = rad;
	}
	if (color) {
		delete[]color;
		color = nullptr;
	}
	QString name = QString::fromStdString(pobj->Name());
	pgds[name].name = name;
	pgds[name].mat = NO_MATERIAL;
	pgds[name].np = pobj->NumParticle();
	pgds[name].cnp = pobj->NumCluster();
	pgds[name].min_rad = pobj->MinRadius();
	pgds[name].max_rad = pobj->MaxRadius();
	np = np + pobj->NumParticle();
	color = new float[np * 4];
	
	return _define();
}

bool xvParticle::defineFromViewFile(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	int ns = 0;
	QString name;
	unsigned int _np = 0;
	qf.read((char*)&_np, sizeof(unsigned int));
	double* d_pos = NULL;
	double* d_mass = NULL;
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
		qf.read((char*)&(pgds[name].cnp), sizeof(unsigned int));
		qf.read((char*)&(pgds[name].min_rad), sizeof(double));
		qf.read((char*)&(pgds[name].max_rad), sizeof(double));
		resizePositionMemory(np, np + pgds[name].np);
		d_pos = new double[pgds[name].np * 4];
		d_mass = new double[pgds[name].cnp ? pgds[name].cnp : pgds[name].np];
		qf.read((char*)d_pos, sizeof(double) * pgds[name].np * 4);
		for (unsigned int i = 0; i < pgds[name].np; i++)
		{
			pos[(np + i) * 4 + 0] = (float)d_pos[i * 4 + 0];
			pos[(np + i) * 4 + 1] = (float)d_pos[i * 4 + 1];
			pos[(np + i) * 4 + 2] = (float)d_pos[i * 4 + 2];
			double rad = pos[(np + i) * 4 + 3] = (float)d_pos[i * 4 + 3];
			if (rad < min_radius)
				min_radius = rad;
			if (rad > max_radius)
				max_radius = rad;
		}
		qf.read((char*)d_mass, sizeof(double) * (pgds[name].cnp ? pgds[name].cnp : pgds[name].np));
		np += pgds[name].np;
		delete[] d_pos;
		delete[] d_mass;
		delete[] _name;
	}
	color = new float[np * 4];
	return _define();
}

bool xvParticle::defineFromListFile(QString path)
{
	QFile qf(path);
	qf.open(QIODevice::ReadOnly);
	QString name;
	QTextStream qts(&qf);
	qts >> name;
	unsigned int _np = 0;
	qts >> _np;
	resizePositionMemory(np, np + _np);
	double d_pos[4] = { 0, };
	for (unsigned int i = 0; i < _np; i++)
	{
		qts >> d_pos[0] >> d_pos[1] >> d_pos[2] >> d_pos[3];
		pos[(np + i) * 4 + 0] = (float)d_pos[0];
		pos[(np + i) * 4 + 1] = (float)d_pos[1];
		pos[(np + i) * 4 + 2] = (float)d_pos[2];
		double rad = pos[(np + i) * 4 + 3] = (float)d_pos[3];
		if (rad < min_radius)
			min_radius = rad;
		if (rad > max_radius)
			max_radius = rad;
	}
	np += _np;
	
	color = new float[np * 4];
	return _define();
}

void xvParticle::setRelativePosition(unsigned int sz, double* d, double* r)
{
	r_pos = new double[sz * 4];
	for (unsigned int i = 0; i < sz; i++)
	{
		r_pos[i * 4 + 0] = d[i * 3 + 0];
		r_pos[i * 4 + 1] = d[i * 3 + 1];
		r_pos[i * 4 + 2] = d[i * 3 + 2];
		r_pos[i * 4 + 3] = r[i];
	}
	r_np = sz;
}

void xvParticle::setParticlePosition(double* p, unsigned int n)
{
// 	if (pos && p)
// 		memcpy(pos, p, sizeof(double) * n * 4);
}

bool xvParticle::UploadParticleFromFile(unsigned int i, QString path)
{
	////qDebug() << i;
	//double ct = 0.0;
	//unsigned int neach = 1;
	//unsigned int inp = 0;
	//unsigned int ins = 0;
	//unsigned int sid = 0;
	//unsigned int vid = 0;
	//QFile qf(path);
	//qf.open(QIODevice::ReadOnly);
	//qf.read((char*)&ct, sizeof(double));
	//qf.read((char*)&inp, sizeof(unsigned int));
	//qf.read((char*)&ins, sizeof(unsigned int));
	//if (np != inp)
	//{
	//	return false;
	//}
	//if (inp != ins)
	//	neach = inp / ins;
	//sid = inp * i * 4;
	//vid = inp * i * 3;
	//double *_pos = new double[inp * 4];
	//double *_vel = new double[ins * 3];
	//qf.read((char*)_pos, sizeof(double) * inp * 4);
	//qf.read((char*)_vel, sizeof(double) * ins * 3);
	//qf.close();
	//xvAnimationController::addTime(i, ct);
	//for (unsigned int j = 0; j < inp; j++)
	//{
	//	unsigned int s = j * 4;
	//	
	//	vector4f p = new_vector4f(
	//		static_cast<float>(_pos[s + 0]),
	//		static_cast<float>(_pos[s + 1]),
	//		static_cast<float>(_pos[s + 2]),
	//		static_cast<float>(_pos[s + 3]));

	//	s += sid;
	//	buffers[s + 0] = p.x;// static_cast<float>(_pos[s + 0]);
	//	buffers[s + 1] = p.y;// static_cast<float>(_pos[s + 1]);
	//	buffers[s + 2] = p.z;// static_cast<float>(_pos[s + 2]);
	//	buffers[s + 3] = p.w;// static_cast<float>(_pos[s + 3]);
	//	if (max_position[0] < p.x) max_position[0] = p.x;// buffers[s + 0];
	//	if (max_position[1] < p.y) max_position[1] = p.y;// buffers[s + 1];
	//	if (max_position[2] < p.z) max_position[2] = p.z;// buffers[s + 2];

	//	if (min_position[0] > p.x) min_position[0] = p.x;// buffers[s + sid + 0];
	//	if (min_position[1] > p.y) min_position[1] = p.y;// buffers[s + sid + 1];
	//	if (min_position[2] > p.z) min_position[2] = p.z;// buffers[s + sid + 2];
	//	float p_mag = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
	//	if (max_position_mag < p_mag) max_position_mag = p_mag;
	//	if (min_position_mag > p_mag) min_position_mag = p_mag;
	//	vector3f vv = new_vector3f(0, 0, 0);
	//	unsigned int v = j * 3;
	//	if (inp != ins)
	//		v = (j / neach) * 3;
	//	vv = new_vector3f(
	//		static_cast<float>(_vel[v + 0]),
	//		static_cast<float>(_vel[v + 1]),
	//		static_cast<float>(_vel[v + 2]));

	//	v = j * 3 + vid;
	//	
	//	vbuffers[v + 0] = vv.x;// static_cast<float>(_vel[v + 0]);
	//	vbuffers[v + 1] = vv.y;// static_cast<float>(_vel[v + 1]);
	//	vbuffers[v + 2] = vv.z;// static_cast<float>(_vel[v + 2]);

	//	if (max_velocity[0] < vv.x) max_velocity[0] = vv.x;// vbuffers[v + vid + 0];
	//	if (max_velocity[1] < vv.y) max_velocity[1] = vv.y;// vbuffers[v + vid + 1];
	//	if (max_velocity[2] < vv.z) max_velocity[2] = vv.z;// vbuffers[v + vid + 2];

	//	if (min_velocity[0] > vv.x) min_velocity[0] = vv.x;// vbuffers[v + vid + 0];
	//	if (min_velocity[1] > vv.y) min_velocity[1] = vv.y;// vbuffers[v + vid + 1];
	//	if (min_velocity[2] > vv.z) min_velocity[2] = vv.z;// vbuffers[v + vid + 2];

	//	float v_mag = sqrt(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z);// [v + 0] * vbuffers[v + 0] + vbuffers[v + 1] * vbuffers[v + 1] + vbuffers[v + 2] * vbuffers[v + 2]);
	//	
	//	if (max_velocity_mag < v_mag) max_velocity_mag = v_mag;
	//	if (min_velocity_mag > v_mag) min_velocity_mag = v_mag;
	//	color_buffers[s + 0] = 0.0f;
	//	color_buffers[s + 1] = 0.0f;
	//	color_buffers[s + 2] = 1.0f;
	//	color_buffers[s + 3] = 1.0f;


	//}
	//delete[] _pos;
	//delete[] _vel;
	return true;
}

bool xvParticle::UploadParticleFromRelativePosition(unsigned int i, vector3d & p, euler_parameters & ep)
{
	//matrix33d A = GlobalTransformationMatrix(ep);
	//unsigned int sid = np * i * 4;
	//unsigned int vid = np * i * 3;
	//xvAnimationController::addTime(i, 0);
	//for (unsigned int j = 0; j < np; j++)
	//{
	//	unsigned int s = j * 4 + sid;
	//	unsigned int v = j * 3 + vid;
	//	vector3d rp = new_vector3d(r_pos[s + 0], r_pos[s + 1], r_pos[s + 2]);
	//	vector3d gp = p + A * rp;
	//	buffers[s + 0] = static_cast<float>(gp.x);
	//	buffers[s + 1] = static_cast<float>(gp.y);
	//	buffers[s + 2] = static_cast<float>(gp.z);
	//	buffers[s + 3] = static_cast<float>(r_pos[s + 3]);

	//	vbuffers[v + 0] = 0.0f;// static_cast<float>(_vel[v + 0]);
	//	vbuffers[v + 1] = 0.0f;//static_cast<float>(_vel[v + 1]);
	//	vbuffers[v + 2] = 0.0f;// static_cast<float>(_vel[v + 2]);

	//	color_buffers[s + 0] = 0.0f;
	//	color_buffers[s + 1] = 0.0f;
	//	color_buffers[s + 2] = 1.0f;
	//	color_buffers[s + 3] = 1.0f;
	//}
	return false;
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

float * xvParticle::ColorBuffers()
{
	return cbuffers;
}

float * xvParticle::PositionBuffers()
{
	return pbuffers;
}

float * xvParticle::VelocityBuffers()
{
	return vbuffers;
}

float xvParticle::getMinValue(xColorControl::ColorMapType cmt)
{
	float v = 0.0;
	/*switch (cmt)
	{
	case xColorControl::COLORMAP_POSITION_X: v = min_position[0]; break;
	case xColorControl::COLORMAP_POSITION_Y: v = min_position[1]; break;
	case xColorControl::COLORMAP_POSITION_Z: v = min_position[2]; break;
	case xColorControl::COLORMAP_VELOCITY_X: v = min_velocity[0]; break;
	case xColorControl::COLORMAP_VELOCITY_Y: v = min_velocity[1]; break;
	case xColorControl::COLORMAP_VELOCITY_Z: v = min_velocity[2]; break;
	case xColorControl::COLORMAP_POSITION_MAG: v = min_position_mag; break;
	case xColorControl::COLORMAP_VELOCITY_MAG: v = min_velocity_mag; break;
	}*/
	return v;
}

float xvParticle::getMaxValue(xColorControl::ColorMapType cmt)
{
	float v = 0.0;
	/*switch (cmt)
	{
	case xColorControl::COLORMAP_POSITION_X: v = max_position[0]; break;
	case xColorControl::COLORMAP_POSITION_Y: v = max_position[1]; break;
	case xColorControl::COLORMAP_POSITION_Z: v = max_position[2]; break;
	case xColorControl::COLORMAP_VELOCITY_X: v = max_velocity[0]; break;
	case xColorControl::COLORMAP_VELOCITY_Y: v = max_velocity[1]; break;
	case xColorControl::COLORMAP_VELOCITY_Z: v = max_velocity[2]; break;
	case xColorControl::COLORMAP_POSITION_MAG: v = max_position_mag; break;
	case xColorControl::COLORMAP_VELOCITY_MAG: v = max_velocity_mag; break;
	}*/
	return v;
}

void xvParticle::updatePosition(std::vector<vector4d>& new_pos)
{
	for (unsigned int i = 0; i < new_pos.size(); i++) {
		vector4d p = new_pos[i];
		pos[i * 4 + 0] = p.x;
		pos[i * 4 + 1] = p.y;
		pos[i * 4 + 2] = p.z;
		pos[i * 4 + 3] = p.w;
	}
}

void xvParticle::setColorFromParticleSize()
{
	if (min_radius == max_radius)
		return;
	for (unsigned int i = 0; i < np; i++) {
		double c[3] = { 0, };
		xColorControl::colorFromData(min_radius, max_radius, pos[i * 4 + 3], c);
		color[i * 4 + 0] = (float)c[0];
		color[i * 4 + 1] = (float)c[1];
		color[i * 4 + 2] = (float)c[2];
		color[i * 4 + 3] = 1.0f;
	}
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
	/*if (buffers)
		delete[] buffers;
	buffers = new float[sz * np * 4];
	if (vbuffers)
		delete[] vbuffers;
	vbuffers = new float[sz * np * 3];
	if (color_buffers)
		delete[] color_buffers;
	color_buffers = new float[sz * np * 4];*/
	
}

QString xvParticle::NameOfGroupData(QString& n)
{
	return pgds[n].name;
}

QString xvParticle::MaterialOfGroupData(QString& n)
{
	return QString::fromStdString(NameOfMaterial(pgds[n].mat));
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

void xvParticle::ChangeColor(unsigned int id, QColor c, QColor& pcolor)
{
	unsigned int cid = id * 4;
	pcolor.setRedF(color[cid + 0]);
	pcolor.setGreenF(color[cid + 1]);
	pcolor.setBlueF(color[cid + 2]);
	color[cid + 0] = c.redF();
	color[cid + 1] = c.greenF();
	color[cid + 2] = c.blueF();
}

void xvParticle::ChangePosition(unsigned int id, double x, double y, double z)
{
	unsigned int cid = id * 4;
	pos[cid + 0] = (float)x;
	pos[cid + 1] = (float)y;
	pos[cid + 2] = (float)z;
}

void xvParticle::MoveParticle(unsigned int id, double x, double y, double z)
{
	pos[id * 4 + 0] += (float)x;
	pos[id * 4 + 1] += (float)y;
	pos[id * 4 + 2] += (float)z;
}

vector3f xvParticle::NormalTwoParticles(unsigned int id, unsigned int jd)
{
	vector3f ip = new_vector3f(pos[id * 4 + 0], pos[id * 4 + 1], pos[id * 4 + 2]);
	vector3f jp = new_vector3f(pos[jd * 4 + 0], pos[jd * 4 + 1], pos[jd * 4 + 2]);
	return normalize(jp - ip);
}

float xvParticle::DistanceTwoParticlesFromSurface(unsigned int id, unsigned int jd)
{
	vector3f ip = new_vector3f(pos[id * 4 + 0], pos[id * 4 + 1], pos[id * 4 + 2]);
	vector3f jp = new_vector3f(pos[jd * 4 + 0], pos[jd * 4 + 1], pos[jd * 4 + 2]);
	return pos[id * 4 + 3] + pos[jd * 4 + 3] - length(jp - ip);
}

bool xvParticle::defineFromRelativePosition(vector3d & p, euler_parameters & ep)
{
	resizePositionMemory(np, np + r_np);
	matrix33d A = GlobalTransformationMatrix(ep);
	for (unsigned int i = 0; i < r_np; i++)
	{
		vector3d rp = new_vector3d(r_pos[i * 4 + 0], r_pos[i * 4 + 1], r_pos[i * 4 + 2]);
		vector3d gp = p + A * rp;
		pos[(np + i) * 4 + 0] = (float)gp.x;
		pos[(np + i) * 4 + 1] = (float)gp.y;
		pos[(np + i) * 4 + 2] = (float)gp.z;
		pos[(np + i) * 4 + 3] = (float)r_pos[i * 4 + 3];
	}
	np += r_np;

	color = new float[np * 4];
	return _define();
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