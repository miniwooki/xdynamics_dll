#ifndef XCOLORCONTROL_H
#define XCOLORCONTROL_H

#include <QList>
#include <QColor>

class xColorControl
{
public:
	enum ColorMapType { COLORMAP_PRESSURE = 0, COLORMAP_POSITION_X, COLORMAP_POSITION_Y, COLORMAP_POSITION_Z, COLORMAP_POSITION_MAG, COLORMAP_VELOCITY_X, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z, COLORMAP_VELOCITY_MAG, COLORMAP_ENUMS/*, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z*/ };

	xColorControl();
	xColorControl(size_t _npart);
	~xColorControl();

	void clearMemory();
	void setColormap(QColor *clist, double *lim);
	void initColormap(size_t npart);
	QColor getColor(int i) { return QColor(c[i][0], c[i][1], c[i][2]); }
	double getLimit(int i);// { return limits[i]; }
	void setTarget(colorMapTarget _cmt) { cmt = _cmt; }
	void setMinMax(size_t cpart, double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8);
	void getColorRamp(size_t c, double v, double *clr);
	void setLimitsFromMinMax();
	void setNumParts(size_t _npart) { npart = _npart; }
	ColorMapType target() { return cmt; }

	static double* particleColorBySphType(unsigned int np, unsigned int* tp);

private:
	static double* tmp_color;
	size_t npart;
	double *limits;
	double c[17][3];
	ColorMapType cmt;

	double *min_vx;
	double *min_vy;
	double *min_vz;
	double *max_vx;
	double *max_vy;
	double *max_vz;
	double *min_p;
	double *max_p;

};

#endif