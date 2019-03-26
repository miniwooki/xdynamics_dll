#ifndef XVMATH_H
#define XYMATH_H

inline void local2global_bryant(double ax, double ay, double az, double sx, double sy, double sz, double* o)
{
	double b[9] = {
		cos(ay) * cos(az), -cos(ay) * sin(az), sin(ay),
		cos(ax) * sin(az) + sin(ax) * sin(ay) * cos(az), cos(ax) * cos(az) - sin(ax) * sin(ay) * sin(az), -sin(ax) * cos(ay),
		sin(ax) * sin(az) - cos(ax) * sin(ay) * cos(az), sin(ax) * cos(az) + cos(ax) * sin(ay) * sin(az), cos(ax) * cos(ay) };
	o[0] = b[0] * sx + b[1] * sy + b[2] * sz;
	o[1] = b[3] * sx + b[4] * sy + b[5] * sz;
	o[2] = b[6] * sx + b[7] * sy + b[8] * sz;
}

#endif