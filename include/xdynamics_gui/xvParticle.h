#ifndef XVPARTICLE_H
#define XVPARTICLE_H

#include "xvGlew.h"
#include "xvAnimationController.h"
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
		double min_rad;
		double max_rad;
	};
public:
	xvParticle();
	~xvParticle();

	void draw(GLenum eMode, int wHeight, int protype, double z);
	
	
	bool defineFromViewFile(QString path);

	void setParticlePosition(double* p, unsigned int n);
	bool UploadParticleFromFile(unsigned int i, QString path);
	void openResultFromFile(unsigned int idx);

	void upParticleScale(double v) { pscale += v; }
	void downParticleScale(double v) { pscale -= v; }
	void setBufferMemories(unsigned int sz);
	QMap<QString, particleGroupData>& ParticleGroupData() { return pgds; }

private:
	bool _define();
	void _drawPoints();
	void resizePositionMemory(unsigned int n0, unsigned int n1);
	bool isDefine;

	unsigned int m_posVBO;
	unsigned int m_colorVBO;
	unsigned int m_program;

	unsigned int np;
	float *buffer;
	float *buffers;
	float *vbuffers;
	float *color_buffers;
	float *color_buffer;
	float *pos;
	float *color;

	bool isSetColor;

	float pscale;
	QMap<QString, particleGroupData> pgds;
	shaderProgram program;
};

#endif