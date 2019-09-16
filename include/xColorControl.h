#ifndef XCOLORCONTROL_H
#define XCOLORCONTROL_H

#include <iostream>

class  xColorControl
{
public:
	enum ColorMapType { COLORMAP_NO_TYPE = 0, COLORMAP_POSITION_MAG, COLORMAP_POSITION_X, COLORMAP_POSITION_Y, COLORMAP_POSITION_Z, COLORMAP_VELOCITY_MAG, COLORMAP_VELOCITY_X, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z, COLORMAP_ENUMS/*, COLORMAP_VELOCITY_Y, COLORMAP_VELOCITY_Z*/ };

	xColorControl();
	//xColorControl(size_t _npart);
	~xColorControl();

	//void clearMemory();
	//void setColormap(QColor *clist, double *lim);
	//void initColormap(double lmin, double lmax);
	//QColor getColor(int i) { return QColor(c[i][0], c[i][1], c[i][2]); }
	//double getLimit(int i);// { return limits[i]; }
	static void setTarget(ColorMapType _cmt);// { cmt = _cmt; }
	static ColorMapType Target();
	static bool isUserLimitInput();
	static void setMinMax(float lmin, float lmax);
	void setLimitArray();
	void getColorRamp(float* p, float* v, float* c);
	//void setLimitsFromMinMax();
	//void setNumParts(size_t _npart) { npart = _npart; }
	//ColorMapType target() { return cmt; }

	//static double* particleColorBySphType(unsigned int np, unsigned int* tp);
	static void setUserLimitInputType(bool isu);
	static float minimumLimit();
	static float maximumLimit();

private:
	//static double* tmp_color;
	static bool is_user_limit_input;
	//size_t npart;
	float limits[16];
	float c[17][3];
	static float min_v;
	static float max_v;
	static float range;
	static float dv;
	static ColorMapType cmt;

	//double *min_vx;
	//double *min_vy;
	//double *min_vz;
	//double *max_vx;
	//double *max_vy;
	//double *max_vz;
	//double *min_p;
	//double *max_p;

};

#endif