#ifndef XVPARTICLE_H
#define XVPARTICLE_H

#include "xvGlew.h"
#include "xvAnimationController.h"
#include "xdynamics_algebra/xAlgebraType.h"
#include "xColorControl.h"
#include <QMap>
#include <QFile>

class xParticleObject;

class xvParticle : public xvGlew
{
public:
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
	xvParticle();
	~xvParticle();

	void draw(GLenum eMode, int wHeight, int protype, double z);
	
	void bind_result_buffers(float* pos_buffer, float* vel_buffer, float* color_buffer);
	bool defineFromViewFile(QString path);
	bool defineFromListFile(QString path);
	bool defineFromRelativePosition(vector3d& p, euler_parameters& ep);
	bool defineFromParticleObject(xParticleObject* pobj);
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
	void ChangeColor(unsigned int id, QColor rgb, QColor& pcolor);
	void ChangePosition(unsigned int id, double x, double y, double z);
	void MoveParticle(unsigned int id, double x, double y, double z);
	vector3f NormalTwoParticles(unsigned int id, unsigned int jd);
	float DistanceTwoParticlesFromSurface(unsigned int id, unsigned int jd);
	bool hasRelativePosition() { return r_pos != NULL; }
	float* ColorBuffers();
	float* PositionBuffers();
	float* VelocityBuffers();
	float getMinValue(xColorControl::ColorMapType cmt);
	float getMaxValue(xColorControl::ColorMapType cmt);
	void updatePosition(std::vector<vector4d>& new_pos);

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
	float *pbuffers;
	float *vbuffers;
	float *cbuffers;
	float *pos;
	float *color;
	double *r_pos;
	bool isSetColor;
	
	float pscale;
	QMap<QString, particleGroupData> pgds;
	shaderProgram program;
};

#endif