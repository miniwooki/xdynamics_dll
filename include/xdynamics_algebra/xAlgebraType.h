#ifndef XALGEBRA_TYPE_H
#define XALGEBRA_TYPE_H

#include "xdynamics_decl.h"

// Define vector structure
typedef struct { int x, y; }vector2i;
typedef struct { unsigned int x, y; }vector2ui;
typedef struct { double x, y; }vector2d;
typedef struct { int x, y, z; }vector3i;
typedef struct { unsigned int x, y, z; }vector3ui;
typedef struct { float x, y, z; }vector3f;
typedef struct { double x, y, z; }vector3d;

typedef struct { int x, y, z, w; }vector4i;
typedef struct { unsigned int x, y, z, w; }vector4ui;
typedef struct { float x, y, z, w; }vector4f;
typedef struct { double x, y, z, w; }vector4d;

typedef struct { double e0, e1, e2, e3; }euler_parameters;

typedef struct { double a00, a01, a02, a10, a11, a12, a20, a21, a22; }matrix33d;
typedef struct { double a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23; }matrix34d;
typedef struct { double a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32; }matrix43d;
typedef struct { double a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33; }matrix44d;

class XDYNAMICS_API xMatrixD
{
public:
	xMatrixD();
	xMatrixD(unsigned int r, unsigned int c);
	~xMatrixD();
	void alloc(unsigned int r, unsigned int c);
	double& operator()(const unsigned int r, const unsigned int c);
	double* Data() const;
	void zeros();
	void insert(unsigned int sr, unsigned int sc, matrix34d& m3x4);
	void insert(unsigned int sr, unsigned int sc, vector4d& v4);
	void insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4);
	void plus(unsigned int sr, unsigned int sc, matrix44d& m4x4);
	void plus(unsigned int sr, unsigned int sc, matrix33d& m3x3, bool ist = false);
	void plus(unsigned int sr, unsigned int sc, matrix34d& m3x4, bool ist = false);
	void plus(unsigned int sr, unsigned int sc, matrix43d& m4x3, bool ist = false);

	void columnSwap(unsigned int i, unsigned int j, int* idx);

	unsigned int rows();
	unsigned int cols();

private:
	unsigned int nrow;
	unsigned int ncol;
	double* data;
};

class XDYNAMICS_API xVectorD
{
public:
	xVectorD();
	xVectorD(unsigned int _size);
	~xVectorD();

	double& operator() (const unsigned int idx) const;
	void    operator*=(double v) const;
	void    operator+=(const xVectorD& v) const;
	void alloc(unsigned int _size);
	void zeros();
	void set(double* d);
	unsigned int Size() const;
	double norm();
	double* Data() const;
	void plus(unsigned int s, const vector3d& v3, const vector4d& v4);

private:
	unsigned int size;
	double* data;
};

class XDYNAMICS_API xSparseD
{
public:
	xSparseD();
	//xSparseD(unsigned int _size);
	xSparseD(unsigned int _nr, unsigned int _nc);
	~xSparseD();

	double& operator()(const unsigned int r, const unsigned int c);
	void zeros();
	unsigned int NNZ();
	unsigned int rows();
	unsigned int cols();
	void alloc(unsigned int _nr, unsigned int _nc);
	void insert(unsigned int sr, unsigned int sc, matrix34d& m3x4);
	void insert(unsigned int sr, unsigned int sc, vector4d& v4);
	void insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4);

	unsigned int *ridx;
	unsigned int *cidx;
	double *value;

private:
	unsigned int nr;
	unsigned int nc;
	unsigned int nnz;
	unsigned int size;
};

#endif