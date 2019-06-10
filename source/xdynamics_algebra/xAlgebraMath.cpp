#include "xdynamics_algebra/xAlgebraMath.h"
#include "lapacke.h"

int LinearSolve(int n, int nrhs, xMatrixD& a, int lda, xVectorD& b, int ldb)
{
	lapack_int info;
	lapack_int* ipiv = new lapack_int[n];
	info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, a.Data(), lda, ipiv, b.Data(), ldb);
	delete[] ipiv;
	return info;
}

void coordinatePartitioning(xSparseD& lhs, int* uID)
{
		//std::cout << m;
	xMatrixD m;
	m.alloc(lhs.rows(), lhs.cols());
	for (unsigned int i = 0; i < lhs.NNZ(); i++)
	{
		m(lhs.ridx[i], lhs.cidx[i]) = lhs.value[i];
	}
	int *v = new int[lhs.cols()];
	for (unsigned int i = 0; i < lhs.cols(); i++) v[i] = i;
	int index = 0;
	double base = 0;
	double lamda = 0;
	double lower = 0;
	bool check = false;
	int ndof = lhs.cols() - lhs.rows();
	for (unsigned int i = 0; i < m.rows(); i++)
	{
		index = i;
		base = abs(m(i, index));
		check = 0;
		check = false;
		for (unsigned j(i + 1); j < m.cols(); j++)
		{
			double c = abs(m(i, j));
			if (base < c)
			{
				index = j;
				base = abs(m(i, j));
				check = true;
			}
		}
		if (check) m.columnSwap(i, index, v);
		lamda = 1.0 / m(i, i);
		for (unsigned int j = i; j < m.cols(); j++)
		{
			m(i, j) *= lamda;
		}
		for (unsigned j(i + 1); j < m.rows(); j++)
		{
			lamda = m(j, i) / m(i, i);
			for (unsigned k(i); k < m.cols(); k++)
				m(j, k) = m(j, k) - lamda * m(i, k);
		}
	}
	// separation
	index = 0;
	for (unsigned int i = m.rows(); i < m.cols(); i++)
		uID[index++] = v[i];
	delete[] v;
	// sort
	int buf2, buf1, id = 0;

	for (int i = 0; i < ndof - 1; i++)
	{
		buf1 = uID[i];
		id = i;
		for (int j = i + 1; j < ndof; j++)
		{
			if (buf1 > uID[j])
			{
				buf1 = uID[j];
				id = j;
			}
		}
		buf2 = uID[i];
		uID[i] = uID[id];
		uID[id] = buf2;
	}
}

vector3d EulerParameterToEulerAngle(const euler_parameters& ep)
{
	double m13 = 2.0 * (ep.e1 * ep.e3 + ep.e0 * ep.e2);
	double m23 = 2.0 * (ep.e2 * ep.e3 - ep.e0 * ep.e1);
	double m33 = ep.e0*ep.e0 - ep.e1*ep.e1 - ep.e2*ep.e2 + ep.e3*ep.e3;
	double m31 = 2.0 * (ep.e1 * ep.e3 - ep.e0 * ep.e2);
	double m32 = 2.0 * (ep.e2 * ep.e3 + ep.e0 * ep.e1);
	if (m33 > 1.0)
		m33 = 1.0;
	if (ep.e1 * ep.e2 == 0.0)
	{
		return new_vector3d(xSign(ep.e3) * 2.0 * acos(ep.e0), 0.0, 0.0);
	}
	return new_vector3d(
		atan2(m13, -m23),
		atan2(sqrt(1 - m33 * m33), m33),
		atan2(m31, m32));
}

vector3d ToVector3D(vector3ui& v3)
{
	return
	{
		static_cast<double>(v3.x),
		static_cast<double>(v3.y),
		static_cast<double>(v3.z)
	};
}

vector3ui ToVector3UI(vector3d& v3)
{
	return
	{
		static_cast<unsigned int>(v3.x),
		static_cast<unsigned int>(v3.y),
		static_cast<unsigned int>(v3.z)
	};
}

XDYNAMICS_API vector3i ToVector3I(vector3d& v3)
{
	return
	{
		static_cast<int>(v3.x),
		static_cast<int>(v3.y),
		static_cast<int>(v3.z)
	};
}

int xSign(double v) { return v >= 0 ? 1 : -1; }

// Define vector3 operators
vector3i operator+ (const vector3i &v1, const vector3i &v2) { return vector3i{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
vector3i operator- (const vector3i &v1, const vector3i &v2) { return vector3i{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
vector3i operator* (const int v, const vector3i &v2) { return vector3i{ v * v2.x, v * v2.y, v * v2.z }; }

vector3i operator/ (const vector3i &v1, const int v) { return vector3i{ v1.x / v, v1.y / v, v1.z / v }; }
void operator+= (vector3i &v1, const vector3i &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
void operator-= (vector3i &v1, const vector3i &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }

vector3ui operator+ (const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
vector3ui operator- (const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
vector3ui operator* (const unsigned int v, const vector3ui &v2) { return vector3ui{ v * v2.x, v * v2.y, v * v2.z }; }
vector3ui operator/ (const vector3ui &v1, const unsigned int v) { return vector3ui{ v1.x / v, v1.y / v, v1.z / v }; }
void operator+= (vector3ui &v1, const vector3ui &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
void operator-= (vector3ui &v1, const vector3ui &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }

vector3f operator+ (const vector3f &v1, const vector3f &v2) { return vector3f{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
vector3f operator- (const vector3f &v1, const vector3f &v2) { return vector3f{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
vector3f operator* (const float v, const vector3f &v2) { return vector3f{ v * v2.x, v * v2.y, v * v2.z }; }
vector3f operator/ (const vector3f &v1, const float v) { return vector3f{ v1.x / v, v1.y / v, v1.z / v }; }
void operator+= (vector3f &v1, const vector3f &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
void operator-= (vector3f &v1, const vector3f &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
vector3f operator- (const vector3f& v1) { return vector3f{ -v1.x, -v1.y, -v1.z }; };

vector3d operator+ (const vector3d &v1, const vector3d &v2) { return vector3d{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; }
vector3d operator- (const vector3d &v1, const vector3d &v2) { return vector3d{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; }
vector3d operator* (const double v, const vector3d &v2) { return vector3d{ v * v2.x, v * v2.y, v * v2.z }; }
vector3d operator/ (const vector3d &v1, const double v) { return vector3d{ v1.x / v, v1.y / v, v1.z / v }; }
void operator+= (vector3d &v1, const vector3d &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z; }
void operator-= (vector3d &v1, const vector3d &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z; }
vector3d operator- (const vector3d& v1) { return vector3d{ -v1.x, -v1.y, -v1.z }; }
vector3d operator~ (const vector3d &v1) { return v1; }
bool operator<=(const vector3d& a, const vector3d& b) { return (a.x <= b.x && a.y <= b.y && a.z <= b.z); }
bool operator>=(const vector3d& a, const vector3d& b) { return (a.x >= b.x && a.y >= b.y && a.z >= b.z); }

// Define vector4 operators
vector4i operator+ (const vector4i &v1, const vector4i &v2) { return vector4i{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
vector4i operator- (const vector4i &v1, const vector4i &v2) { return vector4i{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
vector4i operator* (const int v, const vector4i &v2) { return vector4i{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
vector4i operator/ (const vector4i &v1, const int v) { return vector4i{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
void operator+= (vector4i &v1, const vector4i &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
void operator-= (vector4i &v1, const vector4i &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }

vector4ui operator+ (const vector4ui &v1, const vector4ui &v2) { return vector4ui{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
vector4ui operator- (const vector4ui &v1, const vector4ui &v2) { return vector4ui{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
vector4ui operator* (const unsigned int v, const vector4ui &v2) { return vector4ui{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
vector4ui operator/ (const vector4ui &v1, const unsigned int v) { return vector4ui{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
void operator+= (vector4ui &v1, const vector4ui &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
void operator-= (vector4ui &v1, const vector4ui &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }

vector4f operator+ (const vector4f &v1, const vector4f &v2) { return vector4f{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
vector4f operator- (const vector4f &v1, const vector4f &v2) { return vector4f{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
vector4f operator* (const float v, const vector4f &v2) { return vector4f{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
vector4f operator/ (const vector4f &v1, const float v) { return vector4f{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
void operator+= (vector4f &v1, const vector4f &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
void operator-= (vector4f &v1, const vector4f &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }

vector4d operator+ (const vector4d &v1, const vector4d &v2) { return vector4d{ v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w }; }
vector4d operator- (const vector4d &v1, const vector4d &v2) { return vector4d{ v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w }; }
vector4d operator* (const double v, const vector4d &v2) { return vector4d{ v * v2.x, v * v2.y, v * v2.z, v * v2.w }; }
vector4d operator/ (const vector4d &v1, const double v) { return vector4d{ v1.x / v, v1.y / v, v1.z / v, v1.w / v }; }
void operator+= (vector4d &v1, const vector4d &v2) { v1.x += v2.x, v1.y += v2.y, v1.z += v2.z, v1.w += v2.w; }
void operator-= (vector4d &v1, const vector4d &v2) { v1.x -= v2.x, v1.y -= v2.y, v1.z -= v2.z, v1.w -= v2.w; }
vector4d operator~ (const vector4d &v1) { return v1; }
vector4d operator- (const vector4d &v1) { return{ -v1.x, -v1.y, -v1.z, -v1.w }; }

// Define euler_parameters operators
euler_parameters operator+ (const euler_parameters &v1, const euler_parameters &v2) { return euler_parameters{ v1.e0 + v2.e0, v1.e1 + v2.e1, v1.e2 + v2.e2, v1.e3 + v2.e3 }; }
euler_parameters operator- (const euler_parameters &v1, const euler_parameters &v2) { return euler_parameters{ v1.e0 - v2.e0, v1.e1 - v2.e1, v1.e2 - v2.e2, v1.e3 - v2.e3 }; }
vector4d operator* (const double v, const euler_parameters &v2) { return vector4d{ v * v2.e0, v * v2.e1, v * v2.e2, v * v2.e3 }; }
euler_parameters operator/ (const euler_parameters &v1, const double v) { return euler_parameters{ v1.e0 / v, v1.e1 / v, v1.e2 / v, v1.e3 / v }; }
void operator+= (euler_parameters &v1, const euler_parameters &v2) { v1.e0 += v2.e0, v1.e1 += v2.e1, v1.e2 += v2.e2, v1.e3 += v2.e3; }
void operator-= (euler_parameters &v1, const euler_parameters &v2) { v1.e0 -= v2.e0, v1.e1 -= v2.e1, v1.e2 -= v2.e2, v1.e3 -= v2.e3; }

// Define matrix operators
matrix44d operator*(const matrix34d& m4x3, const matrix34d& m3x4)
{
	return matrix44d{
		m4x3.a00 * m3x4.a00 + m4x3.a10 * m3x4.a10 + m4x3.a20 * m3x4.a20, m4x3.a00 * m3x4.a01 + m4x3.a10 * m3x4.a11 + m4x3.a20 * m3x4.a21, m4x3.a00 * m3x4.a02 + m4x3.a10 * m3x4.a12 + m4x3.a20 * m3x4.a22, m4x3.a00 * m3x4.a03 + m4x3.a10 * m3x4.a13 + m4x3.a20 * m3x4.a23,
		m4x3.a01 * m3x4.a00 + m4x3.a11 * m3x4.a10 + m4x3.a21 * m3x4.a20, m4x3.a01 * m3x4.a01 + m4x3.a11 * m3x4.a11 + m4x3.a21 * m3x4.a21, m4x3.a01 * m3x4.a02 + m4x3.a11 * m3x4.a12 + m4x3.a21 * m3x4.a22, m4x3.a01 * m3x4.a03 + m4x3.a11 * m3x4.a13 + m4x3.a21 * m3x4.a23,
		m4x3.a02 * m3x4.a00 + m4x3.a12 * m3x4.a10 + m4x3.a22 * m3x4.a20, m4x3.a02 * m3x4.a01 + m4x3.a12 * m3x4.a11 + m4x3.a22 * m3x4.a21, m4x3.a02 * m3x4.a02 + m4x3.a12 * m3x4.a12 + m4x3.a22 * m3x4.a22, m4x3.a02 * m3x4.a03 + m4x3.a12 * m3x4.a13 + m4x3.a22 * m3x4.a23,
		m4x3.a03 * m3x4.a00 + m4x3.a13 * m3x4.a10 + m4x3.a23 * m3x4.a20, m4x3.a03 * m3x4.a01 + m4x3.a13 * m3x4.a11 + m4x3.a23 * m3x4.a21, m4x3.a03 * m3x4.a02 + m4x3.a13 * m3x4.a12 + m4x3.a23 * m3x4.a22, m4x3.a03 * m3x4.a03 + m4x3.a13 * m3x4.a13 + m4x3.a23 * m3x4.a23
	};
}

matrix34d operator*(const matrix33d& m3x3, const matrix34d& m3x4)
{
	return matrix34d{
		m3x3.a00*m3x4.a00 + m3x3.a01*m3x4.a10 + m3x3.a02*m3x4.a20, m3x3.a00*m3x4.a01 + m3x3.a01*m3x4.a11 + m3x3.a02*m3x4.a21, m3x3.a00*m3x4.a02 + m3x3.a01*m3x4.a12 + m3x3.a02*m3x4.a22, m3x3.a00*m3x4.a03 + m3x3.a01*m3x4.a13 + m3x3.a02*m3x4.a23,
		m3x3.a10*m3x4.a00 + m3x3.a11*m3x4.a10 + m3x3.a12*m3x4.a20, m3x3.a10*m3x4.a01 + m3x3.a11*m3x4.a11 + m3x3.a12*m3x4.a21, m3x3.a10*m3x4.a02 + m3x3.a11*m3x4.a12 + m3x3.a12*m3x4.a22, m3x3.a10*m3x4.a03 + m3x3.a11*m3x4.a13 + m3x3.a12*m3x4.a23,
		m3x3.a20*m3x4.a00 + m3x3.a21*m3x4.a10 + m3x3.a22*m3x4.a20, m3x3.a20*m3x4.a01 + m3x3.a21*m3x4.a11 + m3x3.a22*m3x4.a21, m3x3.a20*m3x4.a02 + m3x3.a21*m3x4.a12 + m3x3.a22*m3x4.a22, m3x3.a20*m3x4.a03 + m3x3.a21*m3x4.a13 + m3x3.a22*m3x4.a23
	};
}

matrix44d operator*(const double m, const matrix44d& m4x4)
{
	matrix44d m4;
	for (int i = 0; i < 16; i++)
	{
		(*(&(m4.a00) + i)) = m * (*((&m4x4.a00) + i));
	}
	return m4;
}

matrix34d operator*(const double m, const matrix34d& m3x4)
{
	matrix34d m34;
	for (int i = 0; i < 12; i++)
	{
		(*(&(m34.a00) + i)) = m * (*((&m3x4.a00) + i));
	}
	return m34;
}

matrix43d operator*(const double m, const matrix43d& m4x3)
{
	matrix43d m43;
	for (int i = 0; i < 12; i++)
	{
		(*(&(m43.a00) + i)) = m * (*((&m4x3.a00) + i));
	}
	return m43;
}

matrix33d operator*(const double m, const matrix33d& m3x3)
{
	matrix33d m33;
	for (int i = 0; i < 9; i++)
	{
		(*(&(m33.a00) + i)) = m * (*((&m3x3.a00) + i));
	}
	return m33;
}

matrix33d operator-(const matrix33d& m3x3)
{
	matrix33d m33;
	for (int i = 0; i < 12; i++)
	{
		(*(&(m33.a00) + i)) = -(*((&m3x3.a00) + i));
	}
	return m33;
}

matrix34d operator-(const matrix34d& m3x4)
{
	matrix34d m34;
	for (int i = 0; i < 12; i++)
	{
		(*(&(m34.a00) + i)) = -(*((&m3x4.a00) + i));
	}
	return m34;
}

matrix43d operator-(const matrix43d& m4x3)
{
	matrix43d m43;
	for (int i = 0; i < 12; i++)
	{
		(*(&(m43.a00) + i)) = -(*((&m4x3.a00) + i));
	}
	return m43;
}

matrix44d operator-(const matrix44d& m4x4)
{
	matrix44d m4;
	for (int i = 0; i < 16; i++)
	{
		(*(&(m4.a00) + i)) = -(*((&m4x4.a00) + i));
	}
	return m4;
}

vector3d operator*(const matrix33d& m3x3, const vector3d& v3)
{
	return vector3d
	{
		m3x3.a00 * v3.x + m3x3.a01 * v3.y + m3x3.a02 * v3.z,
		m3x3.a10 * v3.x + m3x3.a11 * v3.y + m3x3.a12 * v3.z,
		m3x3.a20 * v3.x + m3x3.a21 * v3.y + m3x3.a22 * v3.z
	};
}

vector4d operator*(const matrix44d& m4x4, const vector4d& v4)
{
	return
	{
		m4x4.a00 * v4.x + m4x4.a01 * v4.y + m4x4.a02 * v4.z + m4x4.a03 * v4.w,
		m4x4.a10 * v4.x + m4x4.a11 * v4.y + m4x4.a12 * v4.z + m4x4.a13 * v4.w,
		m4x4.a20 * v4.x + m4x4.a21 * v4.y + m4x4.a22 * v4.z + m4x4.a23 * v4.w,
		m4x4.a30 * v4.x + m4x4.a31 * v4.y + m4x4.a32 * v4.z + m4x4.a33 * v4.w
	};
}

vector4d operator*(const vector3d& v3, const matrix34d& m3x4)
{
	return vector4d
	{
		v3.x * m3x4.a00 + v3.y * m3x4.a10 + v3.z * m3x4.a20,
		v3.x * m3x4.a01 + v3.y * m3x4.a11 + v3.z * m3x4.a21,
		v3.x * m3x4.a02 + v3.y * m3x4.a12 + v3.z * m3x4.a22,
		v3.x * m3x4.a03 + v3.y * m3x4.a13 + v3.z * m3x4.a23,
	};
}

vector3d operator*(const matrix34d& m3x4, const euler_parameters& e)
{
	return vector3d
	{
		m3x4.a00*e.e0 + m3x4.a01*e.e1 + m3x4.a02*e.e2 + m3x4.a03*e.e3,
		m3x4.a10*e.e0 + m3x4.a11*e.e1 + m3x4.a12*e.e2 + m3x4.a13*e.e3,
		m3x4.a20*e.e0 + m3x4.a21*e.e1 + m3x4.a22*e.e2 + m3x4.a23*e.e3
	};
}

vector4d operator*(const matrix44d& m4x4, const euler_parameters& e)
{
	return vector4d
	{
		m4x4.a00*e.e0 + m4x4.a01*e.e1 + m4x4.a02*e.e2 + m4x4.a03*e.e3,
		m4x4.a10*e.e0 + m4x4.a11*e.e1 + m4x4.a12*e.e2 + m4x4.a13*e.e3,
		m4x4.a20*e.e0 + m4x4.a21*e.e1 + m4x4.a22*e.e2 + m4x4.a23*e.e3,
		m4x4.a30*e.e0 + m4x4.a31*e.e1 + m4x4.a32*e.e2 + m4x4.a33*e.e3
	};
}

vector3d operator*(const matrix34d& m3x4, const vector4d& v)
{
	return vector3d
	{
		m3x4.a00*v.x + m3x4.a01*v.y + m3x4.a02*v.z + m3x4.a03*v.w,
		m3x4.a10*v.x + m3x4.a11*v.y + m3x4.a12*v.z + m3x4.a13*v.w,
		m3x4.a20*v.x + m3x4.a21*v.y + m3x4.a22*v.z + m3x4.a23*v.w
	};
}

matrix44d operator+(const matrix44d& a44, const matrix44d& b44)
{
	return matrix44d
	{
		a44.a00 + b44.a00, a44.a01 + b44.a01, a44.a02 + b44.a02, a44.a03 + b44.a03,
		a44.a10 + b44.a10, a44.a11 + b44.a11, a44.a12 + b44.a12, a44.a13 + b44.a13,
		a44.a20 + b44.a20, a44.a21 + b44.a21, a44.a22 + b44.a22, a44.a23 + b44.a23,
		a44.a30 + b44.a30, a44.a31 + b44.a31, a44.a32 + b44.a32, a44.a33 + b44.a33
	};
}

matrix44d operator-(const matrix44d& a44, const matrix44d& b44)
{
	return matrix44d
	{
		a44.a00 - b44.a00, a44.a01 - b44.a01, a44.a02 - b44.a02, a44.a03 - b44.a03,
		a44.a10 - b44.a10, a44.a11 - b44.a11, a44.a12 - b44.a12, a44.a13 - b44.a13,
		a44.a20 - b44.a20, a44.a21 - b44.a21, a44.a22 - b44.a22, a44.a23 - b44.a23,
		a44.a30 - b44.a30, a44.a31 - b44.a31, a44.a32 - b44.a32, a44.a33 - b44.a33
	};
}

matrix34d operator+(const matrix34d& a34, const matrix34d& b34)
{
	return 
	{
		a34.a00 + b34.a00, a34.a01 + b34.a01, a34.a02 + b34.a02, a34.a03 + b34.a03,
		a34.a10 + b34.a10, a34.a11 + b34.a11, a34.a12 + b34.a12, a34.a13 + b34.a13,
		a34.a20 + b34.a20, a34.a21 + b34.a21, a34.a22 + b34.a22, a34.a23 + b34.a23,
	};
}

matrix33d operator+(const matrix33d& a33, const matrix33d& b33)
{
	return 
	{
		a33.a00 + b33.a00, a33.a01 + b33.a01, a33.a02 + b33.a02,
		a33.a10 + b33.a10, a33.a11 + b33.a11, a33.a12 + b33.a12,
		a33.a20 + b33.a20, a33.a21 + b33.a21, a33.a22 + b33.a22
	};
}

matrix34d operator-(const matrix34d& m1, const matrix34d& m2)
{
	return
	{
		m1.a00 - m2.a00, m1.a01 - m2.a01, m1.a02 - m2.a02, m1.a03 - m2.a03,
		m1.a10 - m2.a10, m1.a11 - m2.a11, m1.a12 - m2.a12, m1.a13 - m2.a13,
		m1.a20 - m2.a20, m1.a21 - m2.a21, m1.a22 - m2.a22, m1.a23 - m2.a23
	};
}

vector4d operator* (const matrix34d& m3x4, const vector3d& v3)
{
	return
	{
		m3x4.a00 * v3.x + m3x4.a10 * v3.y + m3x4.a20 * v3.z,
		m3x4.a01 * v3.x + m3x4.a11 * v3.y + m3x4.a21 * v3.z,
		m3x4.a02 * v3.x + m3x4.a12 * v3.y + m3x4.a22 * v3.z,
		m3x4.a03 * v3.x + m3x4.a13 * v3.y + m3x4.a23 * v3.z,
	};
}

matrix33d operator*(const vector3d& v3, const vector3d& v3t)
{
	return 
	{
		v3.x * v3t.x, v3.x * v3t.y, v3.x * v3t.z,
		v3.y * v3t.x, v3.y * v3t.y, v3.y * v3t.z,
		v3.z * v3t.x, v3.z * v3t.y, v3.z * v3t.z
	};
}

matrix33d operator/(const matrix33d& m3x3, const double v)
{
	double d = 1.0 / v;
	return
	{
		d * m3x3.a00, d * m3x3.a01, d * m3x3.a02,
		d * m3x3.a10, d * m3x3.a11, d * m3x3.a12,
		d * m3x3.a20, d * m3x3.a21, d * m3x3.a22
	};
}

matrix43d operator* (const matrix34d& m4x3, const matrix33d& m3x3)
{
	return 
	{
		m4x3.a00 * m3x3.a00 + m4x3.a10 * m3x3.a10 + m4x3.a20 * m3x3.a20, m4x3.a00 * m3x3.a01 + m4x3.a10 * m3x3.a11 + m4x3.a20 * m3x3.a21, m4x3.a00 * m3x3.a02 + m4x3.a10 * m3x3.a12 + m4x3.a20 * m3x3.a22,
		m4x3.a01 * m3x3.a00 + m4x3.a11 * m3x3.a10 + m4x3.a21 * m3x3.a20, m4x3.a01 * m3x3.a01 + m4x3.a11 * m3x3.a11 + m4x3.a21 * m3x3.a21, m4x3.a01 * m3x3.a02 + m4x3.a11 * m3x3.a12 + m4x3.a21 * m3x3.a22,
		m4x3.a02 * m3x3.a00 + m4x3.a12 * m3x3.a10 + m4x3.a22 * m3x3.a20, m4x3.a02 * m3x3.a01 + m4x3.a12 * m3x3.a11 + m4x3.a22 * m3x3.a21, m4x3.a02 * m3x3.a02 + m4x3.a12 * m3x3.a12 + m4x3.a22 * m3x3.a22,
		m4x3.a03 * m3x3.a00 + m4x3.a13 * m3x3.a10 + m4x3.a23 * m3x3.a20, m4x3.a03 * m3x3.a01 + m4x3.a13 * m3x3.a11 + m4x3.a23 * m3x3.a21, m4x3.a03 * m3x3.a02 + m4x3.a13 * m3x3.a12 + m4x3.a23 * m3x3.a22
	};
}

matrix44d operator* (const matrix43d& m4x3, const matrix34d& m3x4)
{
	return
	{
		m4x3.a00 * m3x4.a00 + m4x3.a01 * m3x4.a10 + m4x3.a02 * m3x4.a20, m4x3.a00 * m3x4.a01 + m4x3.a01 * m3x4.a11 + m4x3.a02 * m3x4.a21, m4x3.a00 * m3x4.a02 + m4x3.a01 * m3x4.a12 + m4x3.a02 * m3x4.a22, m4x3.a00 * m3x4.a03 + m4x3.a01 * m3x4.a13 + m4x3.a02 * m3x4.a23,
		m4x3.a10 * m3x4.a00 + m4x3.a11 * m3x4.a10 + m4x3.a12 * m3x4.a20, m4x3.a10 * m3x4.a01 + m4x3.a11 * m3x4.a11 + m4x3.a12 * m3x4.a21, m4x3.a10 * m3x4.a02 + m4x3.a11 * m3x4.a12 + m4x3.a12 * m3x4.a22, m4x3.a10 * m3x4.a03 + m4x3.a11 * m3x4.a13 + m4x3.a12 * m3x4.a23,
		m4x3.a20 * m3x4.a00 + m4x3.a21 * m3x4.a10 + m4x3.a22 * m3x4.a20, m4x3.a20 * m3x4.a01 + m4x3.a21 * m3x4.a11 + m4x3.a22 * m3x4.a21, m4x3.a20 * m3x4.a02 + m4x3.a21 * m3x4.a12 + m4x3.a22 * m3x4.a22, m4x3.a20 * m3x4.a03 + m4x3.a21 * m3x4.a13 + m4x3.a22 * m3x4.a23,
		m4x3.a30 * m3x4.a00 + m4x3.a31 * m3x4.a10 + m4x3.a32 * m3x4.a20, m4x3.a30 * m3x4.a01 + m4x3.a31 * m3x4.a11 + m4x3.a32 * m3x4.a21, m4x3.a30 * m3x4.a02 + m4x3.a31 * m3x4.a12 + m4x3.a32 * m3x4.a22, m4x3.a30 * m3x4.a03 + m4x3.a31 * m3x4.a13 + m4x3.a32 * m3x4.a23,
	};
}

// Define dot product
int dot(const vector3i &v1, const vector3i &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
unsigned int dot(const vector3ui &v1, const vector3ui &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
float dot(const vector3f &v1, const vector3f &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
double dot(const vector3d &v1, const vector3d &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

int dot(const vector4i &v1, const vector4i &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
unsigned int dot(const vector4ui &v1, const vector4ui &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
float dot(const vector4f &v1, const vector4f &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }
double dot(const vector4d &v1, const vector4d &v2) { return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w; }

double dot(const euler_parameters& e1, const euler_parameters& e2) { return e1.e0 * e2.e0 + e1.e1 * e2.e1 + e1.e2 * e2.e2 + e1.e3 * e2.e3; }
double dot(const euler_parameters& e, const vector4d& v4){ return e.e0 * v4.x + e.e1 * v4.y + e.e2 * v4.z + e.e3 * v4.w; }
double dot(const vector4d& v4, const euler_parameters& e){ return v4.x * e.e0 + v4.y * e.e1 + v4.z * e.e2 + v4.w * e.e3; }

// Define cross product
vector3i cross(const vector3i &v1, const vector3i &v2) { return vector3i{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
vector3ui cross(const vector3ui &v1, const vector3ui &v2) { return vector3ui{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
vector3f cross(const vector3f &v1, const vector3f &v2) { return vector3f{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }
vector3d cross(const vector3d &v1, const vector3d &v2) { return vector3d{ v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x }; }

// Define length product
double length(const vector3i &v) { return sqrt(dot(v, v)); }
double length(const vector3ui &v) { return sqrt(dot(v, v)); }
double length(const vector3f &v) { return sqrt(dot(v, v)); }
double length(const vector3d &v) { return sqrt(dot(v, v)); }
double length(const vector4d &v){ return sqrt(dot(v, v)); }

vector3d normalize(const vector3d& v)
{
	return v / length(v);
}

vector4d normalize(const vector4d & v)
{
	return v / length(v);
}

double xmin(double v1, double v2, double v3)
{
	return v1 < v2 ? (v1 < v3 ? v1 : (v3 < v2 ? v3 : v2)) : (v2 < v3 ? v2 : v3);
}

double xmax(double v1, double v2, double v3)
{
	return v1 > v2 ? (v1 > v3 ? v1 : (v3 > v2 ? v3 : v2)) : (v2 > v3 ? v2 : v3);
}

void inverse(matrix33d& A)
{
	double det = A.a00*A.a11 - A.a01*A.a10 - A.a00*A.a21 + A.a01*A.a20 + A.a10*A.a21 - A.a11*A.a20;
	A =
	{
		(A.a11*A.a22 - A.a12*A.a21) / det, -(A.a01*A.a22 - A.a02*A.a21) / det, (A.a01*A.a12 - A.a02*A.a11) / det,
		-(A.a10*A.a22 - A.a12*A.a20) / det, (A.a00*A.a22 - A.a02*A.a20) / det, -(A.a00*A.a12 - A.a02*A.a10) / det,
		(A.a10*A.a21 - A.a11*A.a20) / det, -(A.a00*A.a21 - A.a01*A.a20) / det, (A.a00*A.a11 - A.a01*A.a10) / det
	};
}

// Define new vectors
vector2i new_vector2i(int x, int y) { return vector2i{ x, y }; }
vector3i new_vector3i(int x, int y, int z) { return vector3i{ x, y, z }; }
vector3ui new_vector3ui(unsigned int x, unsigned int y, unsigned int z) { return vector3ui{ x, y, z }; }
vector3f new_vector3f(float x, float y, float z) { return vector3f{ x, y, z }; }
vector3d new_vector3d(double x, double y, double z) { return vector3d{ x, y, z }; }

vector4i new_vector4i(int x, int y, int z, int w) { return vector4i{ x, y, z, w }; }
vector4ui new_vector4ui(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { return vector4ui{ x, y, z, w }; }
vector4f new_vector4f(float x, float y, float z, float w) { return vector4f{ x, y, z, w }; }
vector4d new_vector4d(double x, double y, double z, double w) { return vector4d{ x, y, z, w }; }

euler_parameters new_euler_parameters(double e0, double e1, double e2, double e3) { return euler_parameters{ e0, e1, e2, e3 }; }

euler_parameters new_euler_parameters(vector4d &e)
{
	return new_euler_parameters(e.x, e.y, e.z, e.w);
}

// matrixd new_matrix(unsigned int nr, unsigned int nc)
// {
// // 	matrixd m;
// // 	m.nrow = nr;
// // 	m.ncol = nc;
// // 	m.data = new double[nr * nc];
// // 	memset(m.data, 0, sizeof(double) * nr * nc);
// // 	return m;
// }

matrix34d GMatrix(const euler_parameters& e)
{
	return matrix34d{
		-e.e1, e.e0, e.e3, -e.e2,
		-e.e2, -e.e3, e.e0, e.e1,
		-e.e3, e.e2, -e.e1, e.e0
	};
}

matrix34d GMatrix(const vector4d& e)
{
	return matrix34d{
		-e.y, e.x, e.w, -e.z,
		-e.z, -e.w, e.x, e.y,
		-e.w, e.z, -e.y, e.x
	};
}

matrix34d LMatrix(const euler_parameters& e)
{
	return matrix34d{
		-e.e1, e.e0, -e.e3, e.e2,
		-e.e2, e.e3, e.e0, -e.e1,
		-e.e3, -e.e2, e.e1, e.e0
	};
}

matrix34d LMatrix(const vector4d & e)
{
	return matrix34d{
		-e.y, e.x, -e.w, e.z,
		-e.z, e.w, e.x, -e.y,
		-e.w, -e.z, e.y, e.x
	};
}

matrix34d BMatrix(const euler_parameters& e, const vector3d& s)
{
	return matrix34d{
		2 * (2 * s.x*e.e0 + e.e2*s.z - e.e3*s.y), 2 * (2 * s.x*e.e1 + e.e3*s.z + e.e2*s.y), 2 * (e.e1*s.y + e.e0*s.z), 2 * (e.e1*s.z - e.e0*s.y),
		2 * (2 * s.y*e.e0 - e.e1*s.z + e.e3*s.x), 2 * (s.x*e.e2 - e.e0*s.z), 2 * (2 * s.y*e.e2 + e.e3*s.z + e.e1*s.x), 2 * (e.e2*s.z + e.e0*s.x),
		2 * (2 * s.z*e.e0 - e.e2*s.x + e.e1*s.y), 2 * (s.x*e.e3 + e.e0*s.y), 2 * (e.e3*s.y - e.e0*s.x), 2 * (2 * s.z*e.e3 + e.e2*s.y + e.e1*s.x)
	};
}

matrix33d GlobalTransformationMatrix(const euler_parameters& e)
{
	matrix33d A;
	A.a00 = 2 * (e.e0*e.e0 + e.e1*e.e1 - 0.5);	A.a01 = 2 * (e.e1*e.e2 - e.e0*e.e3);		A.a02 = 2 * (e.e1*e.e3 + e.e0*e.e2);
	A.a10 = 2 * (e.e1*e.e2 + e.e0*e.e3);		A.a11 = 2 * (e.e0*e.e0 + e.e2*e.e2 - 0.5);	A.a12 = 2 * (e.e2*e.e3 - e.e0*e.e1);
	A.a20 = 2 * (e.e1*e.e3 - e.e0*e.e2);		A.a21 = 2 * (e.e2*e.e3 + e.e0*e.e1);		A.a22 = 2 * (e.e0*e.e0 + e.e3*e.e3 - 0.5);
	return A;
}

matrix33d DGlobalTransformationMatrix(const euler_parameters& e, const euler_parameters& ev)
{
	matrix34d dL = LMatrix(ev);
	matrix34d G = 2.0 * GMatrix(e);
	return
	{
		dL.a00 * G.a00 + dL.a01 * G.a01 + dL.a02 * G.a02 + dL.a03 * G.a03, dL.a00 * G.a10 + dL.a01 * G.a11 + dL.a02 * G.a12 + dL.a03 * G.a13, dL.a00 * G.a20 + dL.a01 * G.a21 + dL.a02 * G.a22 + dL.a03 * G.a23,
		dL.a10 * G.a00 + dL.a11 * G.a01 + dL.a12 * G.a02 + dL.a13 * G.a03, dL.a10 * G.a10 + dL.a11 * G.a11 + dL.a12 * G.a12 + dL.a13 * G.a13, dL.a10 * G.a20 + dL.a11 * G.a21 + dL.a12 * G.a22 + dL.a13 * G.a23,
		dL.a20 * G.a00 + dL.a21 * G.a01 + dL.a22 * G.a02 + dL.a23 * G.a03, dL.a20 * G.a10 + dL.a21 * G.a11 + dL.a22 * G.a12 + dL.a23 * G.a13, dL.a20 * G.a20 + dL.a21 * G.a21 + dL.a22 * G.a22 + dL.a23 * G.a23
	};
}

matrix44d MMatrix(const vector3d& v)
{
	return matrix44d
	{
		0.0, -v.x, -v.y, -v.z,
		v.x, 0.0, v.z, -v.y,
		v.y, -v.z, 0.0, v.x,
		v.z, v.y, -v.x, 0.0
	};
}

matrix44d DMatrix(const vector3d& s, const vector3d& d)
{
	return matrix44d
	{
		4.0 * (s.x*d.x + s.y*d.y + s.z*d.z), 2.0 * (s.y*d.z - s.z*d.y), 2.0 * (s.z*d.x - s.x*d.z), 2.0 * (s.x*d.y - s.y*d.x),
		2.0 * (s.y*d.z - s.z*d.y), 4.0 * s.x*d.x, 2.0 * (s.x*d.y + s.y*d.x), 2.0 * (s.x*d.z + s.z*d.x),
		2.0 * (s.z*d.x - s.x*s.z), 2.0 * (s.x*d.y + s.y*d.x), 4.0 * s.y*d.y, 2.0 * (s.y*d.z + s.z*d.y),
		2.0 * (s.x*d.y - s.y*d.x), 2.0 * (s.x*d.z + s.z*d.x), 2.0 * (s.y*d.z + s.z*d.y), 4.0 * s.z*d.z
	};
}

matrix44d Inverse4X4(const matrix44d& A)
{
	double det =
		A.a00*A.a11*A.a22*A.a33 + A.a00*A.a12*A.a23*A.a31 + A.a00*A.a13*A.a21*A.a32 -
		A.a00*A.a13*A.a22*A.a31 - A.a00*A.a12*A.a21*A.a33 - A.a00*A.a11*A.a23*A.a32 -
		A.a01*A.a10*A.a22*A.a33 - A.a02*A.a10*A.a23*A.a31 - A.a03*A.a10*A.a21*A.a32 +
		A.a03*A.a10*A.a22*A.a31 + A.a02*A.a10*A.a21*A.a33 + A.a01*A.a10*A.a23*A.a32 +
		A.a01*A.a12*A.a20*A.a33 + A.a02*A.a13*A.a20*A.a31 + A.a03*A.a11*A.a20*A.a32 -
		A.a03*A.a12*A.a20*A.a31 - A.a02*A.a11*A.a20*A.a33 - A.a01*A.a13*A.a20*A.a32 -
		A.a01*A.a12*A.a23*A.a30 - A.a02*A.a13*A.a21*A.a30 - A.a03*A.a11*A.a22*A.a30 +
		A.a03*A.a12*A.a21*A.a30 + A.a02*A.a11*A.a23*A.a30 + A.a01*A.a13*A.a22*A.a30;
	matrix44d o;
	o.a00 =  A.a11*A.a22*A.a33 + A.a12*A.a23*A.a31 + A.a13*A.a21*A.a32 - A.a13*A.a22*A.a31 - A.a12*A.a21*A.a33 - A.a11*A.a23*A.a32;
	o.a01 = -A.a01*A.a22*A.a33 - A.a02*A.a23*A.a31 - A.a03*A.a21*A.a32 + A.a03*A.a22*A.a31 + A.a02*A.a21*A.a33 + A.a01*A.a23*A.a32;
	o.a02 =  A.a01*A.a12*A.a33 + A.a02*A.a13*A.a31 + A.a03*A.a11*A.a32 - A.a03*A.a12*A.a31 - A.a02*A.a11*A.a33 - A.a01*A.a13*A.a32;
	o.a03 = -A.a01*A.a12*A.a23 - A.a02*A.a13*A.a21 - A.a03*A.a11*A.a22 + A.a03*A.a12*A.a21 + A.a02*A.a11*A.a23 + A.a01*A.a13*A.a22;

	o.a10 = -A.a10*A.a22*A.a33 - A.a12*A.a23*A.a30 - A.a13*A.a20*A.a32 + A.a13*A.a22*A.a30 + A.a12*A.a20*A.a33 + A.a10*A.a23*A.a32;
	o.a11 = A.a00*A.a22*A.a33 + A.a02*A.a23*A.a30 + A.a03*A.a20*A.a32 - A.a03*A.a22*A.a30 - A.a02*A.a20*A.a33 - A.a00*A.a23*A.a32;
	o.a12 = -A.a00*A.a12*A.a33 - A.a02*A.a13*A.a30 - A.a03*A.a10*A.a32 + A.a03*A.a12*A.a30 + A.a02*A.a10*A.a33 + A.a00*A.a13*A.a32;
	o.a13 = A.a00*A.a12*A.a23 + A.a02*A.a13*A.a20 + A.a03*A.a10*A.a22 - A.a03*A.a12*A.a20 - A.a02*A.a10*A.a23 - A.a00*A.a13*A.a22;

	o.a20 = A.a10*A.a21*A.a33 + A.a11*A.a23*A.a30 + A.a13*A.a20*A.a31 - A.a13*A.a21*A.a30 - A.a11*A.a20*A.a33 - A.a10*A.a23*A.a31;
	o.a21 = -A.a00*A.a21*A.a33 - A.a01*A.a23*A.a30 - A.a03*A.a20*A.a31 + A.a03*A.a21*A.a30 + A.a01*A.a20*A.a33 + A.a00*A.a23*A.a31;
	o.a22 = A.a00*A.a11*A.a33 + A.a01*A.a13*A.a30 + A.a03*A.a10*A.a31 - A.a03*A.a11*A.a30 - A.a01*A.a10*A.a33 - A.a00*A.a13*A.a31;
	o.a23 = -A.a00*A.a11*A.a23 - A.a01*A.a13*A.a20 - A.a03*A.a10*A.a21 + A.a03*A.a11*A.a20 + A.a01*A.a10*A.a23 + A.a00*A.a13*A.a21;

	o.a30 = -A.a10*A.a21*A.a32 - A.a11*A.a22*A.a30 - A.a12*A.a20*A.a31 + A.a12*A.a21*A.a30 + A.a11*A.a20*A.a32 + A.a10*A.a22*A.a31;
	o.a31 = A.a00*A.a21*A.a32 + A.a01*A.a22*A.a30 + A.a02*A.a20*A.a31 - A.a02*A.a21*A.a30 - A.a01*A.a20*A.a32 - A.a00*A.a22*A.a31;
	o.a32 = -A.a00*A.a11*A.a32 - A.a01*A.a12*A.a30 - A.a02*A.a10*A.a31 + A.a02*A.a11*A.a30 + A.a01*A.a10*A.a32 + A.a00*A.a12*A.a31;
	o.a33 = A.a00*A.a11*A.a22 + A.a01*A.a12*A.a20 + A.a02*A.a10*A.a21 - A.a02*A.a11*A.a20 - A.a01*A.a10*A.a22 - A.a00*A.a12*A.a21;
	return (1.0 / det) * o;
}

matrix33d new_identity3(const double j)
{
	return { j, 0, 0, 0, j, 0, 0, 0, j };
}

vector3d ToAngularVelocity(const euler_parameters& e, const euler_parameters& ev)
{
	return 2.0 * GMatrix(e) * ev;
}

vector3d ToGlobal(const euler_parameters& e, const vector3d& v3)
{
	vector3d tv;
	tv = GlobalTransformationMatrix(e) * v3;
// 	tv.x = A.a00*v.x + A.a01*v.y + A.a02*v.z;
// 	tv.y = A.a10*v.x + A.a11*v.y + A.a12*v.z;
// 	tv.z = A.a20*v.x + A.a21*v.y + A.a22*v.z;
	return tv;
}

vector3d ToLocal(const euler_parameters & e, const vector3d & v3)
{
	vector3d tv;
	tv = Transpose(GlobalTransformationMatrix(e)) * v3;
	return tv;
}

matrix33d Tilde(const vector3d & v)
{
	matrix33d m;
	m.a00 = 0.0; m.a01 = -v.z; m.a02 = v.y;
	m.a10 = v.z; m.a11 = 0.0; m.a12 = -v.x;
	m.a20 = -v.y; m.a21 = v.x; m.a22 = 0.0;
	return m;
}

matrix33d Transpose(const matrix33d& m3x3)
{
	matrix33d A;
	A.a00 = m3x3.a00; A.a01 = m3x3.a10; A.a02 = m3x3.a20;
	A.a10 = m3x3.a01; A.a11 = m3x3.a11; A.a12 = m3x3.a21;
	A.a20 = m3x3.a02; A.a21 = m3x3.a12; A.a22 = m3x3.a22;
	return A;
}