#include "xColorControl.h"
//#include "vcontroller.h"

//using namespace ucolors;

//double* xColorControl::tmp_color = NULL;
bool xColorControl::is_user_limit_input = false;
float xColorControl::min_v = 0.0f;
float xColorControl::max_v = 0.0f;
float xColorControl::range = 0.0f;
float xColorControl::dv = 0.0f;
xColorControl::ColorMapType xColorControl::cmt = xColorControl::COLORMAP_VELOCITY_MAG;

xColorControl::xColorControl()
{
	c[0][0] = 0; c[0][1] = 0, c[0][2] = 255;
	c[1][0] = 0; c[1][1] = 63, c[1][2] = 255;
	c[2][0] = 0; c[2][1] = 127, c[2][2] = 255;
	c[3][0] = 0; c[3][1] = 191, c[3][2] = 255;
	c[4][0] = 0; c[4][1] = 255, c[4][2] = 255;
	c[5][0] = 0; c[5][1] = 255, c[5][2] = 191;
	c[6][0] = 0; c[6][1] = 255, c[6][2] = 127;
	c[7][0] = 0; c[7][1] = 255, c[7][2] = 63;
	c[8][0] = 0; c[8][1] = 255, c[8][2] = 0;
	c[9][0] = 63; c[9][1] = 255, c[9][2] = 0;
	c[10][0] = 127; c[10][1] = 255, c[10][2] = 0;
	c[11][0] = 191; c[11][1] = 255, c[11][2] = 0;
	c[12][0] = 255; c[12][1] = 255, c[12][2] = 0;
	c[13][0] = 255; c[13][1] = 191, c[13][2] = 0;
	c[14][0] = 255; c[14][1] = 127, c[14][2] = 0;
	c[15][0] = 255; c[15][1] = 63, c[15][2] = 0;
	c[16][0] = 255; c[16][1] = 0, c[16][2] = 0;
}

xColorControl::~xColorControl()
{
	//clearMemory();
}

void xColorControl::setUserLimitInputType(bool isu)
{
	is_user_limit_input = isu;
}

void xColorControl::setTarget(ColorMapType _cmt)
{
	cmt = _cmt;
}

xColorControl::ColorMapType xColorControl::Target()
{
	return cmt;
}

bool xColorControl::isUserLimitInput()
{
	return is_user_limit_input;
}

void xColorControl::setMinMax(float lmin, float lmax)
{
	min_v = lmin;
	max_v = lmax;
	range = lmax - lmin;
	dv = range / 18.0f;
}

void xColorControl::setLimitArray()
{
	for (int i = 1; i < 17; i++)
	{
		limits[i - 1] = min_v + dv * i;
	}
}

void xColorControl::getColorRamp(float* p, float* v, float* clr)
{
	float div = 1.0f / 255.f;
	float d = 0.f;
	int t = 0;
	switch (cmt)
	{
	case COLORMAP_NO_TYPE: return;
	case COLORMAP_POSITION_MAG: d = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); break;
	case COLORMAP_POSITION_X:   d = p[0]; break;
	case COLORMAP_POSITION_Y:	d = p[1]; break;
	case COLORMAP_POSITION_Z:	d = p[2]; break;
	case COLORMAP_VELOCITY_MAG:	d = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); break;
	case COLORMAP_VELOCITY_X:	d = v[0]; break;
	case COLORMAP_VELOCITY_Y:	d = v[1]; break;
	case COLORMAP_VELOCITY_Z:	d = v[2]; break;
	}
	if (d <= limits[0])
	{
		clr[0] = c[0][0] * div; clr[1] = c[0][1] * div; clr[2] = c[0][2] * div;
		return;
	}
	else if (d >= limits[15])
	{
		clr[0] = c[16][0] * div; clr[1] = c[16][1] * div; clr[2] = c[0][2] * div;
		return;
	}
	for (int i = 1; i < 16; i++) {
		if (d <= limits[i]) {
			t = i;
			break;
		}
	}
	clr[0] = c[t][0] * div; clr[1] = c[t][1] * div; clr[2] = c[t][2] * div;
}
