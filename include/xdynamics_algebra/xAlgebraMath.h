#ifndef XALGEBRA_MATH_H
#define XALGEBRA_MATH_H

#include "xdynamics_algebra/xAlgebraType.h"

#define xsign(a) a >= 0 ? 1 : -1

inline double frand() { return rand() / (double)RAND_MAX; }

// Declaration vector3i operators
XDYNAMICS_API vector3i operator+ (const vector3i &v1, const vector3i &v2);
XDYNAMICS_API vector3i operator- (const vector3i &v1, const vector3i &v2);
XDYNAMICS_API vector3i operator* (const int v, const vector3i &v2);
XDYNAMICS_API vector3i operator/ (const vector3i &v1, const int v);
XDYNAMICS_API void operator+= (vector3i &v1, const vector3i &v2);
XDYNAMICS_API void operator-= (vector3i &v1, const vector3i &v2);

// Declaration vector3ui operators
XDYNAMICS_API vector3ui operator+ (const vector3ui &v1, const vector3ui &v2);
XDYNAMICS_API vector3ui operator- (const vector3ui &v1, const vector3ui &v2);
XDYNAMICS_API vector3ui operator* (const unsigned int v, const vector3ui &v2);
XDYNAMICS_API vector3ui operator/ (const vector3ui &v1, const unsigned int v);
XDYNAMICS_API void operator+= (vector3ui &v1, const vector3ui &v2);
XDYNAMICS_API void operator-= (vector3ui &v1, const vector3ui &v2);

// Declaration vector3f operators
XDYNAMICS_API vector3f operator+ (const vector3f &v1, const vector3f &v2);
XDYNAMICS_API vector3f operator- (const vector3f &v1, const vector3f &v2);
XDYNAMICS_API vector3f operator* (const float v, const vector3f &v2);
XDYNAMICS_API vector3f operator/ (const vector3f &v1, const float v);
XDYNAMICS_API void operator+= (vector3f &v1, const vector3f &v2);
XDYNAMICS_API void operator-= (vector3f &v1, const vector3f &v2);
XDYNAMICS_API vector3f operator-(const vector3f &v1);

// Declaration vector3d operators
XDYNAMICS_API vector3d operator+ (const vector3d &v1, const vector3d &v2);
XDYNAMICS_API vector3d operator- (const vector3d &v1, const vector3d &v2);
XDYNAMICS_API vector3d operator* (const double v, const vector3d &v2);
XDYNAMICS_API vector3d operator/ (const vector3d &v1, const double v);
XDYNAMICS_API void operator+= (vector3d &v1, const vector3d &v2);
XDYNAMICS_API void operator-= (vector3d &v1, const vector3d &v2);
XDYNAMICS_API vector3d operator- (const vector3d &v1);
XDYNAMICS_API vector3d operator~ (const vector3d &v1);
XDYNAMICS_API bool operator<=(const vector3d& a, const vector3d& b);
XDYNAMICS_API bool operator>=(const vector3d& a, const vector3d& b);

// Declaration vector4i operators
XDYNAMICS_API vector4i operator+ (const vector4i &v1, const vector4i &v2);
XDYNAMICS_API vector4i operator- (const vector4i &v1, const vector4i &v2);
XDYNAMICS_API vector4i operator* (const int v, const vector4i &v2);
XDYNAMICS_API vector4i operator/ (const vector4i &v1, const int v);
XDYNAMICS_API void operator+= (vector4i &v1, const vector4i &v2);
XDYNAMICS_API void operator-= (vector4i &v1, const vector4i &v2);

// Declaration vector4ui operators
XDYNAMICS_API vector4ui operator+ (const vector4ui &v1, const vector4ui &v2);
XDYNAMICS_API vector4ui operator- (const vector4ui &v1, const vector4ui &v2);
XDYNAMICS_API vector4ui operator* (const unsigned int v, const vector4ui &v2);
XDYNAMICS_API vector4ui operator/ (const vector4ui &v1, const unsigned int v);
XDYNAMICS_API void operator+= (vector4ui &v1, const vector4ui &v2);
XDYNAMICS_API void operator-= (vector4ui &v1, const vector4ui &v2);

// Declaration vector4f operators
XDYNAMICS_API vector4f operator+ (const vector4f &v1, const vector4f &v2);
XDYNAMICS_API vector4f operator- (const vector4f &v1, const vector4f &v2);
XDYNAMICS_API vector4f operator* (const float v, const vector4f &v2);
XDYNAMICS_API vector4f operator/ (const vector4f &v1, const float v);
XDYNAMICS_API void operator+= (vector4f &v1, const vector4f &v2);
XDYNAMICS_API void operator-= (vector4f &v1, const vector4f &v2);

// Declaration vector4d operators
XDYNAMICS_API vector4d operator+ (const vector4d &v1, const vector4d &v2);
XDYNAMICS_API vector4d operator- (const vector4d &v1, const vector4d &v2);
XDYNAMICS_API vector4d operator* (const double v, const vector4d &v2);
XDYNAMICS_API vector4d operator/ (const vector4d &v1, const double v);
XDYNAMICS_API void operator+= (vector4d &v1, const vector4d &v2);
XDYNAMICS_API void operator-= (vector4d &v1, const vector4d &v2);
XDYNAMICS_API vector4d operator~ (const vector4d &v1);
XDYNAMICS_API vector4d operator- (const vector4d &v1);

// Declaration euler parameters operators
XDYNAMICS_API euler_parameters operator+ (const euler_parameters &v1, const euler_parameters &v2);
XDYNAMICS_API euler_parameters operator- (const euler_parameters &v1, const euler_parameters &v2);
XDYNAMICS_API euler_parameters operator* (const double v, const euler_parameters &v2);
XDYNAMICS_API euler_parameters operator/ (const euler_parameters &v1, const double v);
XDYNAMICS_API void operator+= (euler_parameters &v1, const euler_parameters &v2);
XDYNAMICS_API void operator-= (euler_parameters &v1, const euler_parameters &v2);

// Declaration matrix operators
XDYNAMICS_API matrix44d operator*(const matrix34d& m4x3, const matrix34d& m3x4);
XDYNAMICS_API matrix34d operator*(const matrix33d& m3x3, const matrix34d& m3x4);
XDYNAMICS_API matrix44d operator*(const double m, const matrix44d& m4x4);
XDYNAMICS_API matrix34d operator*(const double m, const matrix34d& m3x4);
XDYNAMICS_API matrix43d operator*(const double m, const matrix43d& m4x3);
XDYNAMICS_API matrix33d operator*(const double m, const matrix33d& m3x3);
XDYNAMICS_API matrix33d operator-(const matrix33d& m3x3);
XDYNAMICS_API matrix34d operator-(const matrix34d& m3x4);
XDYNAMICS_API matrix43d operator-(const matrix43d& m4x3);
XDYNAMICS_API matrix44d operator-(const matrix44d& m4x4);
XDYNAMICS_API vector3d operator* (const matrix33d& m3x3, const vector3d& v3);
XDYNAMICS_API vector4d operator* (const vector3d& v3, const matrix34d& m3x4);
XDYNAMICS_API vector3d operator* (const matrix34d& m3x4, const euler_parameters& e);
XDYNAMICS_API vector4d operator* (const matrix44d& m3x4, const euler_parameters& e);
XDYNAMICS_API vector3d operator* (const matrix34d& m3x4, const vector4d& v);
XDYNAMICS_API vector4d operator* (const matrix44d& m4x4, const vector4d& v);
XDYNAMICS_API matrix44d operator+ (const matrix44d& a44, const matrix44d& b44);
XDYNAMICS_API matrix44d operator- (const matrix44d& a44, const matrix44d& b44);
XDYNAMICS_API matrix33d operator+ (const matrix33d& a33, const matrix33d& b33);
XDYNAMICS_API matrix34d operator+ (const matrix34d& a34, const matrix34d& b34);
XDYNAMICS_API matrix34d operator- (const matrix34d& m1, const matrix34d& m2);
XDYNAMICS_API vector4d operator* (const matrix34d& m3x4, const vector3d& v3);
XDYNAMICS_API matrix33d operator* (const vector3d& v3, const vector3d& v3t);
XDYNAMICS_API matrix33d operator/ (const matrix33d& m3x3, const double v);
XDYNAMICS_API matrix43d operator* (const matrix34d& m4x3, const matrix33d& m3x3);
XDYNAMICS_API matrix44d operator* (const matrix43d& m4x3, const matrix34d& m3x4);

// Declaration new vectors
XDYNAMICS_API vector2i new_vector2i(int x, int y);
XDYNAMICS_API vector3i new_vector3i(int x, int y, int z);
XDYNAMICS_API vector3ui new_vector3ui(unsigned int x, unsigned int y, unsigned int z);
XDYNAMICS_API vector3f new_vector3f(float x, float y, float z);
XDYNAMICS_API vector3d new_vector3d(double x, double y, double z);

XDYNAMICS_API vector4i new_vector4i(int x, int y, int z, int w);
XDYNAMICS_API vector4ui new_vector4ui(unsigned int x, unsigned int y, unsigned int z, unsigned int w);
XDYNAMICS_API vector4f new_vector4f(float x, float y, float z, float w);
XDYNAMICS_API vector4d new_vector4d(double x, double y, double z, double w);

XDYNAMICS_API matrix44d new_matrix44d(const matrix34d& m34d, const euler_parameters& e);

XDYNAMICS_API euler_parameters new_euler_parameters(double e0, double e1, double e2, double e3);
XDYNAMICS_API euler_parameters new_euler_parameters(vector4d& e);

//XDYNAMICS_API matrixd new_matrix(unsigned int nr, unsigned int nc);

// Declaration dot product
XDYNAMICS_API int dot(const vector3i &v1, const vector3i &v2);
XDYNAMICS_API unsigned int dot(const vector3ui &v1, const vector3ui &v2);
XDYNAMICS_API float dot(const vector3f &v1, const vector3f &v2);
XDYNAMICS_API double dot(const vector3d &v1, const vector3d &v2);

XDYNAMICS_API int dot(const vector4i &v1, const vector4i &v2);
XDYNAMICS_API unsigned int dot(const vector4ui &v1, const vector4ui &v2);
XDYNAMICS_API float dot(const vector4f &v1, const vector4f &v2);
XDYNAMICS_API double dot(const vector4d &v1, const vector4d &v2);

XDYNAMICS_API double dot(const euler_parameters& e1, const euler_parameters& e2);
XDYNAMICS_API double dot(const euler_parameters& e, const vector4d& v4);
XDYNAMICS_API double dot(const vector4d& v4, const euler_parameters& e);

// Declaration cross product
XDYNAMICS_API vector3i cross(const vector3i &v1, const vector3i &v2);
XDYNAMICS_API vector3ui cross(const vector3ui &v1, const vector3ui &v2);
XDYNAMICS_API vector3f cross(const vector3f &v1, const vector3f &v2);
XDYNAMICS_API vector3d cross(const vector3d &v1, const vector3d &v2);

XDYNAMICS_API double length(const vector3i &v);
XDYNAMICS_API double length(const vector3ui &v);
XDYNAMICS_API double length(const vector3f &v);
XDYNAMICS_API double length(const vector3d &v);
XDYNAMICS_API double length(const vector4d& v);
XDYNAMICS_API double length(const euler_parameters& v);

XDYNAMICS_API vector3d normalize(const vector3d& v);
XDYNAMICS_API vector4d normalize(const vector4d& v);
XDYNAMICS_API euler_parameters normalize(const euler_parameters& v);

XDYNAMICS_API double xmin(double v1, double v2, double v3 = FLT_MAX);
XDYNAMICS_API double xmax(double v1, double v2, double v3 = -FLT_MIN);

//XDYNAMICS_API int xsign(double d);

XDYNAMICS_API void inverse(matrix33d& A);

XDYNAMICS_API matrix34d GMatrix(const euler_parameters& e);
XDYNAMICS_API matrix34d GMatrix(const vector4d& e);
XDYNAMICS_API matrix34d LMatrix(const euler_parameters& e);
XDYNAMICS_API matrix34d LMatrix(const vector4d& e);
XDYNAMICS_API matrix34d BMatrix(const euler_parameters& e, const vector3d& s);
XDYNAMICS_API matrix33d GlobalTransformationMatrix(const euler_parameters& e);
XDYNAMICS_API matrix33d DGlobalTransformationMatrix(const euler_parameters& e, const euler_parameters& ev);
XDYNAMICS_API matrix44d MMatrix(const vector3d& v);
XDYNAMICS_API matrix44d DMatrix(const vector3d& s, const vector3d& d);
XDYNAMICS_API matrix44d Inverse4X4(const matrix44d& A);
XDYNAMICS_API matrix33d new_identity3(const double j);
XDYNAMICS_API vector3d ToAngularVelocity(const euler_parameters& e, const euler_parameters& ev);
XDYNAMICS_API vector3d ToGlobal(const euler_parameters& e, const vector3d& v3);
XDYNAMICS_API vector3d ToLocal(const euler_parameters& e, const vector3d& v3);
XDYNAMICS_API matrix33d Tilde(const vector3d& v);
XDYNAMICS_API matrix33d Transpose(const matrix33d& A);
XDYNAMICS_API int LinearSolve(int n, int nrhs, xMatrixD& a, int lda, xVectorD& b, int ldb);
XDYNAMICS_API void coordinatePartitioning(xSparseD& lhs, int* uID);
XDYNAMICS_API vector3d EulerParameterToEulerAngle(const euler_parameters& e);
XDYNAMICS_API euler_parameters EulerAngleToEulerParameters(const vector3d v3);
XDYNAMICS_API euler_parameters CalculateUCEOM(matrix33d& J, euler_parameters& ep, euler_parameters& ev, vector3d& n_prime);
// Conversion
XDYNAMICS_API vector3d ToVector3D(vector3ui& v3);
XDYNAMICS_API vector3ui ToVector3UI(vector3d& v3);
XDYNAMICS_API vector3i ToVector3I(vector3d& v3);
XDYNAMICS_API vector3d ToVector3D(vector3f& v3);
XDYNAMICS_API euler_parameters ToEulerParameters(vector4d& v4);
XDYNAMICS_API int xSign(double v);

#endif