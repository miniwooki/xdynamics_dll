//xdynamics_manager.cpp : DLL 응용 프로그램을 위해 내보낸 함수를 정의합니다.
//
//
//#include "stdafx.h"
//#include "xdynamics_algebra/xdynamics_algebra.h"
//#include "lapacke.h"
//
//int LinearSolve(int n, int nrhs, xMatrixD& a, int lda, xVectorD& b, int ldb)
//{
// 	lapack_int info;
// 	lapack_int* ipiv = new lapack_int[n];
// 	info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a.Data(), lda, ipiv, b.Data(), ldb);
// 	delete[] ipiv;
//	return info;
//}
//
//vector3d ToVector3D(vector3ui& v3)
//{
//	return
//	{
//		static_cast<double>(v3.x),
//		static_cast<double>(v3.y),
//		static_cast<double>(v3.z)
//	};
//}
//
//vector3ui ToVector3UI(vector3d& v3)
//{
//	return
//	{
//		static_cast<unsigned int>(v3.x),
//		static_cast<unsigned int>(v3.y),
//		static_cast<unsigned int>(v3.z)
//	};
//}
//
//Define vector3 operators
//vector3i operator+ (const vector3i &v1, const vector3i &v2) { return vector3i{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
//vector3i operator- (const vector3i &v1, const vector3i &v2) { return vector3i{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
//vector3i operator* (const int v, const vector3i &v2) { return vector3i{ v * v2.x, v * v2.y, v * v2.z }; }
//vector3i operator/ (const vector3i &v1, const int v) { return vector3i{ v1.x / v, v1.y / v, v1.z / v }; }
//void operator+= (vector3i &v1, const vector3i &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
//void operator-= (vector3i &v1, const vector3i &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
//
//vector3ui operator+ (const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
//vector3ui operator- (const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
//vector3ui operator* (const unsigned int v, const vector3ui &v2) { return vector3ui{ v * v2.x, v * v2.y, v * v2.z }; }
//vector3ui operator/ (const vector3ui &v1, const unsigned int v) { return vector3ui{ v1.x / v, v1.y / v, v1.z / v }; }
//void operator+= (vector3ui &v1, const vector3ui &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
//void operator-= (vector3ui &v1, const vector3ui &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
//
//vector3f operator+ (const vector3f &v1, const vector3f &v2) { return vector3f{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
//vector3f operator- (const vector3f &v1, const vector3f &v2) { return vector3f{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
//vector3f operator* (const float v, const vector3f &v2) { return vector3f{ v * v2.x, v * v2.y, v * v2.z }; }
//vector3f operator/ (const vector3f &v1, const float v) { return vector3f{ v1.x / v, v1.y / v, v1.z / v }; }
//void operator+= (vector3f &v1, const vector3f &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
//void operator-= (vector3f &v1, const vector3f &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
//vector3f operator- (const vector3f& v1) { return vector3f{ -v1.x, -v1.y, -v1.z }; };
//
//vector3d operator+ (const vector3d &v1, const vector3d &v2) { return vector3d{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
//vector3d operator- (const vector3d &v1, const vector3d &v2) { return vector3d{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
//vector3d operator* (const double v, const vector3d &v2) { return vector3d{ v * v2.x, v * v2.y, v * v2.z }; }
//vector3d operator/ (const vector3d &v1, const double v) { return vector3d{ v1.x / v, v1.y / v, v1.z / v }; }
//void operator+= (vector3d &v1, const vector3d &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
//void operator-= (vector3d &v1, const vector3d &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
//vector3d operator- (const vector3d& v1) { return vector3d{ -v1.x, -v1.y, -v1.z }; }
//vector3d operator~ (const vector3d &v1) { return v1; }
//
//Define vector4 operators
//vector4i operator+ (const vector4i &v1, const vector4i &v2) { return vector4i{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
//vector4i operator- (const vector4i &v1, const vector4i &v2) { return vector4i{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
//vector4i operator* (const int v, const vector4i &v2) { return vector4i{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
//vector4i operator/ (const vector4i &v1, const int v) { return vector4i{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
//void operator+= (vector4i &v1, const vector4i &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
//void operator-= (vector4i &v1, const vector4i &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }
//
//vector4ui operator+ (const vector4ui &v1, const vector4ui &v2) { return vector4ui{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
//vector4ui operator- (const vector4ui &v1, const vector4ui &v2) { return vector4ui{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
//vector4ui operator* (const unsigned int v, const vector4ui &v2) { return vector4ui{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
//vector4ui operator/ (const vector4ui &v1, const unsigned int v) { return vector4ui{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
//void operator+= (vector4ui &v1, const vector4ui &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
//void operator-= (vector4ui &v1, const vector4ui &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }
//
//vector4f operator+ (const vector4f &v1, const vector4f &v2) { return vector4f{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
//vector4f operator- (const vector4f &v1, const vector4f &v2) { return vector4f{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
//vector4f operator* (const float v, const vector4f &v2) { return vector4f{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
//vector4f operator/ (const vector4f &v1, const float v) { return vector4f{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
//void operator+= (vector4f &v1, const vector4f &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
//void operator-= (vector4f &v1, const vector4f &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }
//
//vector4d operator+ (const vector4d &v1, const vector4d &v2) { return vector4d{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
//vector4d operator- (const vector4d &v1, const vector4d &v2) { return vector4d{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
//vector4d operator* (const double v, const vector4d &v2) { return vector4d{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
//vector4d operator/ (const vector4d &v1, const double v) { return vector4d{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
//void operator+= (vector4d &v1, const vector4d &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
//void operator-= (vector4d &v1, const vector4d &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }
//vector4d operator~ (const vector4d &v1) { return v1; }
//vector4d operator- (const vector4d &v1) { return { -v1.x, -v1.y, -v1.z, -v1.w }; }
//
//Define euler_parameters operators
//euler_parameters operator+ (const euler_parameters &v1, const euler_parameters &v2) { return euler_parameters{ v1.e0 + v2.e0, v1.e1 + v2.e1, v1.e2 + v2.e2, v1.e3 + v2.e3 }; }
//euler_parameters operator- (const euler_parameters &v1, const euler_parameters &v2) { return euler_parameters{ v1.e0 - v2.e0, v1.e1 - v2.e1, v1.e2 - v2.e2, v1.e3 - v2.e3 }; }
//euler_parameters operator* (const double v, const euler_parameters &v2) { return euler_parameters{ v * v2.e0, v * v2.e1, v * v2.e2, v * v2.e3 }; }
//euler_parameters operator/ (const euler_parameters &v1, const double v) { return euler_parameters{ v1.e0 / v, v1.e1 / v, v1.e2 / v, v1.e3 / v }; }
//void operator+= (euler_parameters &v1, const euler_parameters &v2) { v1.e0 += v2.e0, v1.e1 += v2.e1, v1.e2 += v2.e2, v1.e3 += v2.e3; }
//void operator-= (euler_parameters &v1, const euler_parameters &v2) { v1.e0 -= v2.e0, v1.e1 -= v2.e1, v1.e2 -= v2.e2, v1.e3 -= v2.e3; }
//
//Define matrix operators
//matrix44d operator*(const matrix34d& m4x3, const matrix34d& m3x4)
//{
//	return matrix44d{
//		m4x3.a00 * m3x4.a00 + m4x3.a10 * m3x4.a10 + m4x3.a20 * m3x4.a20, m4x3.a00 * m3x4.a01 + m4x3.a10 * m3x4.a11 + m4x3.a20 * m3x4.a21, m4x3.a00 * m3x4.a02 + m4x3.a10 * m3x4.a12 + m4x3.a20 * m3x4.a22, m4x3.a00 * m3x4.a03 + m4x3.a10 * m3x4.a13 + m4x3.a20 * m3x4.a23,
//		m4x3.a01 * m3x4.a00 + m4x3.a11 * m3x4.a10 + m4x3.a21 * m3x4.a20, m4x3.a01 * m3x4.a01 + m4x3.a11 * m3x4.a11 + m4x3.a21 * m3x4.a21, m4x3.a01 * m3x4.a02 + m4x3.a11 * m3x4.a12 + m4x3.a21 * m3x4.a22, m4x3.a01 * m3x4.a03 + m4x3.a11 * m3x4.a13 + m4x3.a21 * m3x4.a23,
//		m4x3.a02 * m3x4.a00 + m4x3.a12 * m3x4.a10 + m4x3.a22 * m3x4.a20, m4x3.a02 * m3x4.a01 + m4x3.a12 * m3x4.a11 + m4x3.a22 * m3x4.a21, m4x3.a02 * m3x4.a02 + m4x3.a12 * m3x4.a12 + m4x3.a22 * m3x4.a22, m4x3.a02 * m3x4.a03 + m4x3.a12 * m3x4.a13 + m4x3.a22 * m3x4.a23,
//		m4x3.a03 * m3x4.a00 + m4x3.a13 * m3x4.a10 + m4x3.a23 * m3x4.a20, m4x3.a03 * m3x4.a01 + m4x3.a13 * m3x4.a11 + m4x3.a23 * m3x4.a21, m4x3.a03 * m3x4.a02 + m4x3.a13 * m3x4.a12 + m4x3.a23 * m3x4.a22, m4x3.a03 * m3x4.a03 + m4x3.a13 * m3x4.a13 + m4x3.a23 * m3x4.a23
//	};
//}
//
//matrix34d operator*(const matrix33d& m3x3, const matrix34d& m3x4)
//{
//	return matrix34d{
//		m3x3.a00*m3x4.a00 + m3x3.a01*m3x4.a10 + m3x3.a02*m3x4.a20, m3x3.a00*m3x4.a01 + m3x3.a01*m3x4.a11 + m3x3.a02*m3x4.a21, m3x3.a00*m3x4.a02 + m3x3.a01*m3x4.a12 + m3x3.a02*m3x4.a22, m3x3.a00*m3x4.a03 + m3x3.a01*m3x4.a13 + m3x3.a02*m3x4.a23,
//		m3x3.a10*m3x4.a00 + m3x3.a11*m3x4.a10 + m3x3.a12*m3x4.a20, m3x3.a10*m3x4.a01 + m3x3.a11*m3x4.a11 + m3x3.a12*m3x4.a21, m3x3.a10*m3x4.a02 + m3x3.a11*m3x4.a12 + m3x3.a12*m3x4.a22, m3x3.a10*m3x4.a03 + m3x3.a11*m3x4.a13 + m3x3.a12*m3x4.a23,
//		m3x3.a20*m3x4.a00 + m3x3.a21*m3x4.a10 + m3x3.a22*m3x4.a20, m3x3.a20*m3x4.a01 + m3x3.a21*m3x4.a11 + m3x3.a22*m3x4.a21, m3x3.a20*m3x4.a02 + m3x3.a21*m3x4.a12 + m3x3.a22*m3x4.a22, m3x3.a20*m3x4.a03 + m3x3.a21*m3x4.a13 + m3x3.a22*m3x4.a23
//	};
//}
//
//matrix44d operator*(const double m, const matrix44d& m4x4)
//{
//	matrix44d m4;
//	for (int i = 0; i < 16; i++)
//	{
//		(*(&(m4.a00) + i)) = m * (*((&m4x4.a00) + i));
//	}
//	return m4;
//}
//
//matrix34d operator*(const double m, const matrix34d& m3x4)
//{
//	matrix34d m34;
//	for (int i = 0; i < 12; i++)
//	{
//		(*(&(m34.a00) + i)) = m * (*((&m3x4.a00) + i));
//	}
//	return m34;
//}
//
//matrix34d operator-(const matrix34d& m3x4)
//{
//	matrix34d m34;
//	for (int i = 0; i < 12; i++)
//	{
//		(*(&(m34.a00) + i)) = -(*((&m3x4.a00) + i));
//	}
//	return m34;
//}
//
//matrix44d operator-(const matrix44d& m4x4)
//{
//	matrix44d m4;
//	for (int i = 0; i < 16; i++)
//	{
//		(*(&(m4.a00) + i)) = -(*((&m4x4.a00) + i));
//	}
//	return m4;
//}
//
//vector3d operator*(const matrix33d& m3x3, const vector3d& v3)
//{
//	return vector3d
//	{
//		m3x3.a00 * v3.x + m3x3.a01 * v3.y + m3x3.a02 * v3.z,
//		m3x3.a10 * v3.x + m3x3.a11 * v3.y + m3x3.a12 * v3.z,
//		m3x3.a20 * v3.x + m3x3.a21 * v3.y + m3x3.a22 * v3.z
//	};
//}
//
//vector4d operator*(const vector3d& v3, const matrix34d& m3x4)
//{
//	return vector4d
//	{
//		v3.x * m3x4.a00 + v3.y * m3x4.a10 + v3.z * m3x4.a20,
//		v3.x * m3x4.a01 + v3.y * m3x4.a11 + v3.z * m3x4.a21,
//		v3.x * m3x4.a02 + v3.y * m3x4.a12 + v3.z * m3x4.a22,
//		v3.x * m3x4.a03 + v3.y * m3x4.a13 + v3.z * m3x4.a23,
//	};
//}
//
//vector3d operator*(const matrix34d& m3x4, const euler_parameters& e)
//{
//	return vector3d
//	{
//		m3x4.a00*e.e0 + m3x4.a01*e.e1 + m3x4.a02*e.e2 + m3x4.a03*e.e3,
//		m3x4.a10*e.e0 + m3x4.a11*e.e1 + m3x4.a12*e.e2 + m3x4.a13*e.e3,
//		m3x4.a20*e.e0 + m3x4.a21*e.e1 + m3x4.a22*e.e2 + m3x4.a23*e.e3
//	};
//}
//
//vector3d operator*(const matrix34d& m3x4, const vector4d& v)
//{
//	return vector3d
//	{
//		m3x4.a00*v.x + m3x4.a01*v.y + m3x4.a02*v.z + m3x4.a03*v.w,
//		m3x4.a10*v.x + m3x4.a11*v.y + m3x4.a12*v.z + m3x4.a13*v.w,
//		m3x4.a20*v.x + m3x4.a21*v.y + m3x4.a22*v.z + m3x4.a23*v.w
//	};
//}
//
//matrix44d operator+(const matrix44d& a44, const matrix44d& b44)
//{
//	return matrix44d
//	{
//		a44.a00 + b44.a00, a44.a01 + b44.a01, a44.a02 + b44.a02, a44.a03 + b44.a03,
//		a44.a10 + b44.a10, a44.a11 + b44.a11, a44.a12 + b44.a12, a44.a13 + b44.a13,
//		a44.a20 + b44.a20, a44.a21 + b44.a21, a44.a22 + b44.a22, a44.a23 + b44.a23,
//		a44.a30 + b44.a30, a44.a31 + b44.a31, a44.a32 + b44.a32, a44.a33 + b44.a33
//	};
//}
//
//matrix34d operator-(const matrix34d& m1, const matrix34d& m2)
//{
//	return
//	{
//		m1.a00 - m2.a00, m1.a01 - m2.a01, m1.a02 - m2.a02, m1.a03 - m2.a03,
//		m1.a10 - m2.a10, m1.a11 - m2.a11, m1.a12 - m2.a12, m1.a13 - m2.a13,
//		m1.a20 - m2.a20, m1.a21 - m2.a21, m1.a22 - m2.a22, m1.a23 - m2.a23
//	};
//}
//
//vector4d operator* (const matrix34d& m3x4, const vector3d& v3)
//{
//	return 
//	{
//		m3x4.a00 * v3.x + m3x4.a10 * v3.y + m3x4.a20 * v3.z,
//		m3x4.a01 * v3.x + m3x4.a11 * v3.y + m3x4.a21 * v3.z,
//		m3x4.a02 * v3.x + m3x4.a12 * v3.y + m3x4.a22 * v3.z,
//		m3x4.a03 * v3.x + m3x4.a13 * v3.y + m3x4.a23 * v3.z,
//	};
//}
//
//Define dot product
//int dot(const vector3i &v1, const vector3i &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
//unsigned int dot(const vector3ui &v1, const vector3ui &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
//float dot(const vector3f &v1, const vector3f &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
//double dot(const vector3d &v1, const vector3d &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
//
//int dot(const vector4i &v1, const vector4i &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
//unsigned int dot(const vector4ui &v1, const vector4ui &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
//float dot(const vector4f &v1, const vector4f &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
//double dot(const vector4d &v1, const vector4d &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
//
//double dot(const euler_parameters& e1, const euler_parameters& e2) { return e1.e0 * e2.e0 + e1.e1 * e2.e1 + e1.e2 * e2.e2 + e1.e3 * e2.e3; }
//
//Define cross product
//vector3i cross(const vector3i &v1, const vector3i &v2) { return vector3i{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
//vector3ui cross(const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
//vector3f cross(const vector3f &v1, const vector3f &v2) { return vector3f{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
//vector3d cross(const vector3d &v1, const vector3d &v2) { return vector3d{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
//
//Define length product
//double length(const vector3i &v) { return sqrt(dot(v, v)); }
//double length(const vector3ui &v) { return sqrt(dot(v, v)); }
//double length(const vector3f &v) { return sqrt(dot(v, v)); }
//double length(const vector3d &v) { return sqrt(dot(v, v)); }
//
//void inverse(matrix33d& A)
//{
//	double det = A.a00*A.a11 - A.a01*A.a10 - A.a00*A.a21 + A.a01*A.a20 + A.a10*A.a21 - A.a11*A.a20;
//	A = 
//	{
//		(A.a11*A.a22 - A.a12*A.a21) / det, -(A.a01*A.a22 - A.a02*A.a21) / det, (A.a01*A.a12 - A.a02*A.a11) / det,
//		-(A.a10*A.a22 - A.a12*A.a20) / det, (A.a00*A.a22 - A.a02*A.a20) / det, -(A.a00*A.a12 - A.a02*A.a10) / det,
//		(A.a10*A.a21 - A.a11*A.a20) / det, -(A.a00*A.a21 - A.a01*A.a20) / det, (A.a00*A.a11 - A.a01*A.a10) / det
//	};
//}
//
//Define new vectors
//vector3i new_vector3i(int x, int y, int z) { return vector3i{ x, y, z }; }
//vector3ui new_vector3ui(unsigned int x, unsigned int y, unsigned int z) { return vector3ui{ x, y, z }; }
//vector3f new_vector3f(float x, float y, float z) { return vector3f{ x, y, z }; }
//vector3d new_vector3d(double x, double y, double z) { return vector3d{ x, y, z }; }
//
//vector4i new_vector4i(int x, int y, int z, int w) { return vector4i{ x, y, z, w }; }
//vector4ui new_vector4ui(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return vector4ui{ x, y, z, w }; }
//vector4f new_vector4f(float x, float y, float z, float w) { return vector4f{ x, y, z, w }; }
//vector4d new_vector4d(double x, double y, double z, double w) { return vector4d{ x, y, z, w }; }
//
//euler_parameters new_euler_parameters(double e0, double e1, double e2, double e3) { return euler_parameters{ e0, e1, e2, e3 }; }
//
//matrixd new_matrix(unsigned int nr, unsigned int nc)
//{
//// 	matrixd m;
//// 	m.nrow = nr;
//// 	m.ncol = nc;
//// 	m.data = new double[nr * nc];
//// 	memset(m.data, 0, sizeof(double) * nr * nc);
//// 	return m;
//}
//
//matrix34d GMatrix(const euler_parameters& e)
//{
//	return matrix34d{
//		-e.e1, e.e0, e.e3, -e.e2,
//		-e.e2, -e.e3, e.e0, e.e1,
//		-e.e3, e.e2, -e.e1, e.e0 
//	};
//}
//
//XDYNAMICS_API matrix34d LMatrix(const euler_parameters& e)
//{
//	return matrix34d{
//		-e.e1, e.e0, -e.e3, e.e2,
//		-e.e2, e.e3, e.e0, -e.e1,
//		-e.e3, -e.e2, e.e1, e.e0
//	};
//}
//
//matrix34d BMatrix(const euler_parameters& e, const vector3d& s)
//{
//	return matrix34d{
//		2 * (2 * s.x*e.e0 + e.e2*s.z - e.e3*s.y), 2 * (2 * s.x*e.e1 + e.e3*s.z + e.e2*s.y), 2 * (e.e1*s.y + e.e0*s.z), 2 * (e.e1*s.z - e.e0*s.y),
//		2 * (2 * s.y*e.e0 - e.e1*s.z + e.e3*s.x), 2 * (s.x*e.e2 - e.e0*s.z), 2 * (2 * s.y*e.e2 + e.e3*s.z + e.e1*s.x), 2 * (e.e2*s.z + e.e0*s.x),
//		2 * (2 * s.z*e.e0 - e.e2*s.x + e.e1*s.y), 2 * (s.x*e.e3 + e.e0*s.y), 2 * (e.e3*s.y - e.e0*s.x), 2 * (2 * s.z*e.e3 + e.e2*s.y + e.e1*s.x)
//	};
//}
//
//matrix33d GlobalTransformationMatrix(const euler_parameters& e)
//{
//	matrix33d A;
//	A.a00 = 2 * (e.e0*e.e0 + e.e1*e.e1 - 0.5);	A.a01 = 2 * (e.e1*e.e2 - e.e0*e.e3);		A.a02 = 2 * (e.e1*e.e3 + e.e0*e.e2);
//	A.a10 = 2 * (e.e1*e.e2 + e.e0*e.e3);		A.a11 = 2 * (e.e0*e.e0 + e.e2*e.e2 - 0.5);	A.a12 = 2 * (e.e2*e.e3 - e.e0*e.e1);
//	A.a20 = 2 * (e.e1*e.e3 - e.e0*e.e2);		A.a21 = 2 * (e.e2*e.e3 + e.e0*e.e1);		A.a22 = 2 * (e.e0*e.e0 + e.e3*e.e3 - 0.5);
//	return A;
//}
//
//matrix44d MMatrix(const vector3d& v)
//{
//	return matrix44d
//	{
//		0.0, -v.x, -v.y, -v.z,
//		v.x,  0.0,  v.z, -v.y,
//		v.y, -v.z,  0.0,  v.x,
//		v.z,  v.y, -v.x,  0.0
//	};
//}
//
//matrix44d DMatrix(const vector3d& s, const vector3d& d)
//{
//	return matrix44d
//	{
//		4.0 * (s.x*d.x + s.y*d.y + s.z*d.z),	2.0 * (s.y*d.z - s.z*d.y),	2.0 * (s.z*d.x - s.x*d.z),	2.0 * (s.x*d.y - s.y*d.x),
//		2.0 * (s.y*d.z - s.z*d.y),				4.0 * s.x*d.x,				2.0 * (s.x*d.y + s.y*d.x),	2.0 * (s.x*d.z + s.z*d.x),
//		2.0 * (s.z*d.x - s.x*s.z),				2.0 * (s.x*d.y + s.y*d.x),	4.0 * s.y*d.y,				2.0 * (s.y*d.z + s.z*d.y),
//		2.0 * (s.x*d.y - s.y*d.x),				2.0 * (s.x*d.z + s.z*d.x),	2.0 * (s.y*d.z + s.z*d.y),	4.0 * s.z*d.z
//	};
//}
//
