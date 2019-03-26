// #include "xdynamics_algebra/xAlgebraType.h"

//XVectorDataBase *XVectorDataBase::createData()
//{
//	XVectorDataBase *d = new XVectorDataBase;
//	d->size = 0;
//	d->ref = NULL;
//	return d;
//}
//
//void XVectorDataBase::freeData(XVectorDataBase *d)
//{
//	if (d->ref) delete[] d->ref;
//	delete d;
//}

#include "xdynamics_algebra/xAlgebraType.h"

xMatrixD::xMatrixD()
	: nrow(0)
	, ncol(0)
	, data(NULL)
{

}

xMatrixD::xMatrixD(unsigned int r, unsigned int c)
	: nrow(r)
	, ncol(c)
	, data(NULL)
{
	data = new double[nrow * ncol];
	memset(data, 0, sizeof(double) * nrow * ncol);
}

xMatrixD::~xMatrixD()
{
	if (data) delete[] data;
}

void xMatrixD::alloc(unsigned int r, unsigned int c)
{
	if (!data)
	{
		nrow = r;
		ncol = c;
		data = new double[r * c];
		memset(data, 0, sizeof(double) * nrow * ncol);
	}
}

double& xMatrixD::operator()(const unsigned int r, const unsigned int c)
{
	return data[c * nrow + r];
}

double* xMatrixD::Data() const
{
	return data;
}

void xMatrixD::zeros()
{
	if (data)
		memset(data, 0, sizeof(double) * nrow * ncol);
}

void xMatrixD::insert(unsigned int sr, unsigned int sc, matrix34d& m3x4)
{
	unsigned int cnt = 0;
	double* ptr = &m3x4.a00;
	for (unsigned int r(sr); r < sr + 3; r++)
		for (unsigned int c(sc); c < sc + 4; c++)
			data[c * nrow + r] += *(ptr + cnt++);
}

void xMatrixD::insert(unsigned int sr, unsigned int sc, vector4d& v4)
{
	data[(sc + 0) * nrow + sr] = v4.x;
	data[(sc + 1) * nrow + sr] = v4.y;
	data[(sc + 2) * nrow + sr] = v4.z;
	data[(sc + 3) * nrow + sr] = v4.w;
}

void xMatrixD::insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4)
{
	data[(sc + 0) * nrow + sr] = v3.x;
	data[(sc + 1) * nrow + sr] = v3.y;
	data[(sc + 2) * nrow + sr] = v3.z;
	data[(sc + 4) * nrow + sr] = v4.x;
	data[(sc + 5) * nrow + sr] = v4.y;
	data[(sc + 6) * nrow + sr] = v4.z;
	data[(sc + 7) * nrow + sr] = v4.w;
}

void xMatrixD::plus(unsigned int sr, unsigned int sc, matrix44d& m4x4)
{
	unsigned int cnt = 0;
	double* ptr = &m4x4.a00;
	for (unsigned int r(sr); r < sr + 4; r++)
		for (unsigned int c(sc); c < sc + 4; c++)
			data[c * nrow + r] += *(ptr + cnt++);
}

void xMatrixD::plus(unsigned int sr, unsigned int sc, matrix34d& m3x4, bool ist)
{
	unsigned int cnt = 0;
	double* ptr = &m3x4.a00;
	unsigned int _sr = ist ? sc : sr;
	unsigned int _sc = ist ? sr : sc;
	for (unsigned int r(_sr); r < _sr + 3; r++)
		for (unsigned int c(_sc); c < _sc + 4; c++)
			ist ? data[c * nrow + r] += *(ptr + cnt++) : data[r * nrow + c] += *(ptr + cnt++);
}

void xMatrixD::plus(unsigned int sr, unsigned int sc, matrix33d& m3x3, bool ist /*= false*/)
{
	unsigned int cnt = 0;
	double* ptr = &m3x3.a00;
	unsigned int _sr = ist ? sc : sr;
	unsigned int _sc = ist ? sr : sc;
	for (unsigned int r(_sr); r < _sr + 3; r++)
		for (unsigned int c(_sc); c < _sc + 3; c++)
			ist ? data[c * nrow + r] += *(ptr + cnt++) : data[r * nrow + c] += *(ptr + cnt++);
}

void xMatrixD::plus(unsigned int sr, unsigned int sc, matrix43d& m4x3, bool ist /*= false*/)
{
	unsigned int cnt = 0;
	double* ptr = &m4x3.a00;
	unsigned int _sr = ist ? sc : sr;
	unsigned int _sc = ist ? sr : sc;
	for (unsigned int r(_sr); r < _sr + 4; r++)
		for (unsigned int c(_sc); c < _sc + 3; c++)
			ist ? data[c * nrow + r] += *(ptr + cnt++) : data[r * nrow + c] += *(ptr + cnt++);
}

xVectorD::xVectorD()
	: size(0)
	, data(NULL)
{

}

xVectorD::xVectorD(unsigned int _size)
	: size(_size)
	, data(NULL)
{
	data = new double[size];
	memset(data, 0, sizeof(double) * size);
}

xVectorD::~xVectorD()
{
	if (data) delete[] data; data = NULL;
}

double& xVectorD::operator()(const unsigned int idx) const
{
	return data[idx];
}

void xVectorD::operator*=(double v) const
{
	for (unsigned int i = 0; i < size; i++)
		data[i] *= v;
}

void xVectorD::operator+=(const xVectorD& v) const
{
	for (unsigned i(0); i < v.Size(); i++) data[i] += v(i);
}

void xVectorD::alloc(unsigned int _size)
{
	if (!data)
	{
		size = _size;
		data = new double[size];
		memset(data, 0, sizeof(double) * size);
	}
}

void xVectorD::zeros()
{
	memset(data, 0, sizeof(double) * size);
}

unsigned int xVectorD::Size() const
{
	return size;
}

double xVectorD::norm()
{
	double sum = 0;
	for (unsigned int i = 0; i < size; i++)
		sum += data[i] * data[i];
	return sqrt(sum);
}

double* xVectorD::Data() const
{
	return data;
}

void xVectorD::plus(unsigned int s, const vector3d& v3, const vector4d& v4)
{
	data[s + 0] += v3.x;
	data[s + 1] += v3.y;
	data[s + 2] += v3.z;
	data[s + 3] += v4.x;
	data[s + 4] += v4.y;
	data[s + 5] += v4.z;
	data[s + 6] += v4.w;
}

xSparseD::xSparseD()
	: ridx(NULL)
	, cidx(NULL)
	, value(NULL)
	, size(0)
	, nnz(0)
{

}

xSparseD::xSparseD(unsigned int _size)
	: ridx(NULL)
	, cidx(NULL)
	, value(NULL)
	, nnz(0)
	, size(_size)
{
	ridx = new unsigned int[size];
	cidx = new unsigned int[size];
	value = new double[size];
	memset(ridx, 0, sizeof(unsigned int) * size);
	memset(cidx, 0, sizeof(unsigned int) * size);
	memset(value, 0, sizeof(double) * size);
}

xSparseD::~xSparseD()
{
	nnz = 0;
	if (ridx) delete[] ridx; ridx = NULL;
	if (cidx) delete[] cidx; cidx = NULL;
	if (value) delete[] value; value = NULL;
}

double& xSparseD::operator()(const unsigned int r, const unsigned int c)
{
	ridx[nnz] = r;
	cidx[nnz] = c;
	return value[nnz++];
}

void xSparseD::zeros()
{
	nnz = 0;
}

unsigned int xSparseD::NNZ()
{
	return nnz;
}

void xSparseD::alloc(unsigned int _size)
{
	size = _size;
	ridx = new unsigned int[size];
	cidx = new unsigned int[size];
	value = new double[size];
	nnz = 0;
}

void xSparseD::insert(unsigned int sr, unsigned int sc, matrix34d& m3x4)
{
	unsigned int cnt = 0;
	double v = 0.0;
	double* ptr = &m3x4.a00;
	for (unsigned int r(sr); r < sr + 3; r++)
	{
		for (unsigned int c(sc); c < sc + 4; c++)
		{
			v = *(ptr + cnt++);
			if (v) (*this)(r, c) = v;
		}
	}
}

void xSparseD::insert(unsigned int sr, unsigned int sc, vector4d& v4)
{
	unsigned int cnt = 0;
	double v = 0.0;
	double* ptr = &v4.x;
	for (unsigned int i = 0; i < 4; i++)
	{
		v = *(ptr + cnt++);
		if (v) (*this)(sr, sc + i) = v;
	}
}

void xSparseD::insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4)
{
	(*this)(sr, sc + 0) = v3.x;
	(*this)(sr, sc + 1) = v3.y;
	(*this)(sr, sc + 2) = v3.z;
	(*this)(sr, sc + 3) = v4.x;
	(*this)(sr, sc + 4) = v4.y;
	(*this)(sr, sc + 5) = v4.z;
	(*this)(sr, sc + 6) = v4.w;
}
