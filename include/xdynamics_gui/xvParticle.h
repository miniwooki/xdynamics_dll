#ifndef XVPARTICLE_H
#define XVPARTICLE_H

#include "xvGlew.h"
#include "xvAnimationController.h"
#include "xdynamics_algebra/xAlgebraType.h"
#include <QMap>
#include <QFile>

// QT_BEGIN_NAMESPACE
// class QFile;
// QT_END_NAMESPACE

class xvParticle : public xvGlew
{
	struct particleGroupData
	{
		QString name;
		int mat;
		unsigned int sid;
		unsigned int np;
		unsigned int cnp;
		double min_rad;
		double max_rad;
	};
public:
	xvParticle();
	~xvParticle();

	void draw(GLenum eMode, int wHeight, int protype, double z);
	
	
	bool defineFromViewFile(QString path);
	bool defineFromListFile(QString path);
	bool defineFromRelativePosition(vector3d& p, euler_parameters& ep);
	void setRelativePosition(unsigned int sz, double* d, double* r);

	void setParticlePosition(double* p, unsigned int n);
	bool UploadParticleFromFile(unsigned int i, QString path);
	bool UploadParticleFromRelativePosition(unsigned int i, vector3d& p, euler_parameters& ep);
	void openResultFromFile(unsigned int idx);

	void upParticleScale(double v) { pscale += v; }
	void downParticleScale(double v) { pscale -= v; }
	void setBufferMemories(unsigned int sz);
	QString NameOfGroupData(QString& n);
	QString MaterialOfGroupData(QString& n);
	unsigned int NumParticlesOfGroupData(QString& n);
	unsigned int NumParticles();
	double MinRadiusOfGroupData(QString& n);
	double MaxnRadiusOfGroupData(QString& n);
	QMap<QString, particleGroupData>& ParticleGroupData() { return pgds; }

	bool hasRelativePosition() { return r_pos != NULL; }

private:
	bool _define();
	void _drawPoints(float* pos_buffer, float* color_buffer);
	void resizePositionMemory(unsigned int n0, unsigned int n1);
	bool isDefine;

	unsigned int m_posVBO;
	unsigned int m_colorVBO;
	unsigned int m_program;

	unsigned int r_np;
	unsigned int np;
//	unsigned int *np_buffer;
	//float *buffer;
	float *buffers;
	float *vbuffers;
	float *color_buffers;
//	float *color_buffer;
	float *pos;
	float *color;
	double *r_pos;
	bool isSetColor;

	float pscale;
	QMap<QString, particleGroupData> pgds;
	shaderProgram program;
};

#endif