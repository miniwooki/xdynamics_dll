// #ifndef XDYNAMICS_ALGEBRA_H
// #define XDYNAMICS_ALGEBRA_H
// 
// #include "xdynamics_decl.h"
// 
// 
// 
// // Define vector structure
// typedef struct { int x, y, z; }vector3i;
// typedef struct { unsigned int x, y, z; }vector3ui;
// typedef struct { float x, y, z; }vector3f;
// typedef struct { double x, y, z; }vector3d;
// 
// typedef struct { int x, y, z, w; }vector4i;
// typedef struct { unsigned int x, y, z, w; }vector4ui;
// typedef struct { float x, y, z, w; }vector4f;
// typedef struct { double x, y, z, w; }vector4d;
// 
// typedef struct { double e0, e1, e2, e3; }euler_parameters;
// 
// typedef struct { double a00, a01, a02, a10, a11, a12, a20, a21, a22; }matrix33d;
// typedef struct { double a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23; }matrix34d;
// typedef struct { double a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32; }matrix43d;
// typedef struct { double a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33; }matrix44d;
// 
// // template<class T>
// // struct vector
// // {
// // 	virtual void XDAAPIENTRY call(const T d) = 0;
// // 	virtual ~vector(){}
// // };
// // 
// // extern "C" XDYNAMICS_API vector<double>* XDAAPIENTRY xVectorD();
// 
// class XDYNAMICS_API xMatrixD
// {
// public:
// 	xMatrixD();
// 	xMatrixD(unsigned int r, unsigned int c);
// 	~xMatrixD();
// 	void alloc(unsigned int r, unsigned int c);
// 	double& operator()(const unsigned int r, const unsigned int c);
// 	double* Data() const;
// 	void zeros();
// 	void insert(unsigned int sr, unsigned int sc, matrix34d& m3x4);
// 	void insert(unsigned int sr, unsigned int sc, vector4d& v4);
// 	void insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4);
// 	void plus(unsigned int sr, unsigned int sc, matrix44d& m4x4);
// 	void plus(unsigned int sr, unsigned int sc, matrix34d& m3x4, bool ist = false);
// 
// private:
// 	unsigned int nrow;
// 	unsigned int ncol;
// 	double* data;
// };
// 
// class XDYNAMICS_API xVectorD
// {
// public:
// 	xVectorD();
// 	xVectorD(unsigned int _size);
// 	~xVectorD();
// 
// 	double& operator() (const unsigned int idx) const;
// 	void    operator*=(double v) const;
// 	void    operator+=(const xVectorD& v) const;
// 	void alloc(unsigned int _size);
// 	void zeros();
// 	unsigned int Size() const;
// 	double norm();
// 	double* Data() const;
// 
// private:
// 	unsigned int size;
// 	double* data;
// };
// 
// class XDYNAMICS_API xSparseD
// {
// public:
// 	xSparseD();
// 	xSparseD(unsigned int _size);
// 	~xSparseD();
// 
// 	double& operator()(const unsigned int r, const unsigned int c);
// 	void zeros();
// 	unsigned int NNZ();
// 	void alloc(unsigned int _size);
// 	void insert(unsigned int sr, unsigned int sc, matrix34d& m3x4);
// 	void insert(unsigned int sr, unsigned int sc, vector4d& v4);
// 	void insert(unsigned int sr, unsigned int sc, vector3d& v3, vector4d& v4);
// 
// 	unsigned int *ridx;
// 	unsigned int *cidx;
// 	double *value;
// 
// private:
// 	unsigned int nnz;
// 	unsigned int size;
// };
// 
// //typedef XDYNAMICS_API xMatrixD matrixd;
// /*extern "C" XDYNAMICS_API matrixd;*/
// 
// // Declaration vector3i operators
// XDYNAMICS_API vector3i operator+ (const vector3i &v1, const vector3i &v2);
// XDYNAMICS_API vector3i operator- (const vector3i &v1, const vector3i &v2);
// XDYNAMICS_API vector3i operator* (const int v, const vector3i &v2);
// XDYNAMICS_API vector3i operator/ (const vector3i &v1, const int v);
// XDYNAMICS_API void operator+= (vector3i &v1, const vector3i &v2);
// XDYNAMICS_API void operator-= (vector3i &v1, const vector3i &v2);
// 
// // Declaration vector3ui operators
// XDYNAMICS_API vector3ui operator+ (const vector3ui &v1, const vector3ui &v2);
// XDYNAMICS_API vector3ui operator- (const vector3ui &v1, const vector3ui &v2);
// XDYNAMICS_API vector3ui operator* (const unsigned int v, const vector3ui &v2);
// XDYNAMICS_API vector3ui operator/ (const vector3ui &v1, const unsigned int v);
// XDYNAMICS_API void operator+= (vector3ui &v1, const vector3ui &v2);
// XDYNAMICS_API void operator-= (vector3ui &v1, const vector3ui &v2);
// 
// // Declaration vector3f operators
// XDYNAMICS_API vector3f operator+ (const vector3f &v1, const vector3f &v2);
// XDYNAMICS_API vector3f operator- (const vector3f &v1, const vector3f &v2);
// XDYNAMICS_API vector3f operator* (const float v, const vector3f &v2);
// XDYNAMICS_API vector3f operator/ (const vector3f &v1, const float v);
// XDYNAMICS_API void operator+= (vector3f &v1, const vector3f &v2);
// XDYNAMICS_API void operator-= (vector3f &v1, const vector3f &v2);
// XDYNAMICS_API vector3f operator-(const vector3f &v1);
// 
// // Declaration vector3d operators
// XDYNAMICS_API vector3d operator+ (const vector3d &v1, const vector3d &v2);
// XDYNAMICS_API vector3d operator- (const vector3d &v1, const vector3d &v2);
// XDYNAMICS_API vector3d operator* (const double v, const vector3d &v2);
// XDYNAMICS_API vector3d operator/ (const vector3d &v1, const double v);
// XDYNAMICS_API void operator+= (vector3d &v1, const vector3d &v2);
// XDYNAMICS_API void operator-= (vector3d &v1, const vector3d &v2);
// XDYNAMICS_API vector3d operator- (const vector3d &v1);
// XDYNAMICS_API vector3d operator~ (const vector3d &v1);
// 
// // Declaration vector4i operators
// XDYNAMICS_API vector4i operator+ (const vector4i &v1, const vector4i &v2);
// XDYNAMICS_API vector4i operator- (const vector4i &v1, const vector4i &v2);
// XDYNAMICS_API vector4i operator* (const int v, const vector4i &v2);
// XDYNAMICS_API vector4i operator/ (const vector4i &v1, const int v);
// XDYNAMICS_API void operator+= (vector4i &v1, const vector4i &v2);
// XDYNAMICS_API void operator-= (vector4i &v1, const vector4i &v2);
// 
// // Declaration vector4ui operators
// XDYNAMICS_API vector4ui operator+ (const vector4ui &v1, const vector4ui &v2);
// XDYNAMICS_API vector4ui operator- (const vector4ui &v1, const vector4ui &v2);
// XDYNAMICS_API vector4ui operator* (const unsigned int v, const vector4ui &v2);
// XDYNAMICS_API vector4ui operator/ (const vector4ui &v1, const unsigned int v);
// XDYNAMICS_API void operator+= (vector4ui &v1, const vector4ui &v2);
// XDYNAMICS_API void operator-= (vector4ui &v1, const vector4ui &v2);
// 
// // Declaration vector4f operators
// XDYNAMICS_API vector4f operator+ (const vector4f &v1, const vector4f &v2);
// XDYNAMICS_API vector4f operator- (const vector4f &v1, const vector4f &v2);
// XDYNAMICS_API vector4f operator* (const float v, const vector4f &v2);
// XDYNAMICS_API vector4f operator/ (const vector4f &v1, const float v);
// XDYNAMICS_API void operator+= (vector4f &v1, const vector4f &v2);
// XDYNAMICS_API void operator-= (vector4f &v1, const vector4f &v2);
// 
// // Declaration vector4d operators
// XDYNAMICS_API vector4d operator+ (const vector4d &v1, const vector4d &v2);
// XDYNAMICS_API vector4d operator- (const vector4d &v1, const vector4d &v2);
// XDYNAMICS_API vector4d operator* (const double v, const vector4d &v2);
// XDYNAMICS_API vector4d operator/ (const vector4d &v1, const double v);
// XDYNAMICS_API void operator+= (vector4d &v1, const vector4d &v2);
// XDYNAMICS_API void operator-= (vector4d &v1, const vector4d &v2);
// XDYNAMICS_API vector4d operator~ (const vector4d &v1);
// XDYNAMICS_API vector4d operator- (const vector4d &v1);
// 
// // Declaration euler parameters operators
// XDYNAMICS_API euler_parameters operator+ (const euler_parameters &v1, const euler_parameters &v2);
// XDYNAMICS_API euler_parameters operator- (const euler_parameters &v1, const euler_parameters &v2);
// XDYNAMICS_API euler_parameters operator* (const double v, const euler_parameters &v2);
// XDYNAMICS_API euler_parameters operator/ (const euler_parameters &v1, const double v);
// XDYNAMICS_API void operator+= (euler_parameters &v1, const euler_parameters &v2);
// XDYNAMICS_API void operator-= (euler_parameters &v1, const euler_parameters &v2);
// 
// // Declaration matrix operators
// XDYNAMICS_API matrix44d operator*(const matrix34d& m4x3, const matrix34d& m3x4);
// XDYNAMICS_API matrix34d operator*(const matrix33d& m3x3, const matrix34d& m3x4);
// XDYNAMICS_API matrix44d operator*(const double m, const matrix44d& m4x4);
// XDYNAMICS_API matrix34d operator*(const double m, const matrix34d& m3x4);
// XDYNAMICS_API matrix34d operator-(const matrix34d& m3x4);
// XDYNAMICS_API matrix44d operator-(const matrix44d& m4x4);
// XDYNAMICS_API vector3d operator* (const matrix33d& m3x3, const vector3d& v3);
// XDYNAMICS_API vector4d operator* (const vector3d& v3, const matrix34d& m3x4);
// XDYNAMICS_API vector3d operator* (const matrix34d& m3x4, const euler_parameters& e);
// XDYNAMICS_API vector3d operator* (const matrix34d& m3x4, const vector4d& v);
// XDYNAMICS_API matrix44d operator+ (const matrix44d& a44, const matrix44d& b44);
// XDYNAMICS_API matrix34d operator- (const matrix34d& m1, const matrix34d& m2);
// XDYNAMICS_API vector4d operator* (const matrix34d& m3x4, const vector3d& v3);
// 
// // Declaration new vectors
// XDYNAMICS_API vector3i new_vector3i(int x, int y, int z);
// XDYNAMICS_API vector3ui new_vector3ui(unsigned int x, unsigned int y, unsigned int z);
// XDYNAMICS_API vector3f new_vector3f(float x, float y, float z);
// XDYNAMICS_API vector3d new_vector3d(double x, double y, double z);
// 
// XDYNAMICS_API vector4i new_vector4i(int x, int y, int z, int w);
// XDYNAMICS_API vector4ui new_vector4ui(unsigned int x, unsigned int y, unsigned int z, unsigned int w);
// XDYNAMICS_API vector4f new_vector4f(float x, float y, float z, float w);
// XDYNAMICS_API vector4d new_vector4d(double x, double y, double z, double w);
// 
// XDYNAMICS_API euler_parameters new_euler_parameters(double e0, double e1, double e2, double e3);
// 
// //XDYNAMICS_API matrixd new_matrix(unsigned int nr, unsigned int nc);
// 
// // Declaration dot product
// XDYNAMICS_API int dot(const vector3i &v1, const vector3i &v2);
// XDYNAMICS_API unsigned int dot(const vector3ui &v1, const vector3ui &v2);
// XDYNAMICS_API float dot(const vector3f &v1, const vector3f &v2);
// XDYNAMICS_API double dot(const vector3d &v1, const vector3d &v2);
// 
// XDYNAMICS_API int dot(const vector4i &v1, const vector4i &v2);
// XDYNAMICS_API unsigned int dot(const vector4ui &v1, const vector4ui &v2);
// XDYNAMICS_API float dot(const vector4f &v1, const vector4f &v2);
// XDYNAMICS_API double dot(const vector4d &v1, const vector4d &v2);
// 
// XDYNAMICS_API double dot(const euler_parameters& e1, const euler_parameters& e2);
// 
// // Declaration cross product
// XDYNAMICS_API vector3i cross(const vector3i &v1, const vector3i &v2);
// XDYNAMICS_API vector3ui cross(const vector3ui &v1, const vector3ui &v2);
// XDYNAMICS_API vector3f cross(const vector3f &v1, const vector3f &v2);
// XDYNAMICS_API vector3d cross(const vector3d &v1, const vector3d &v2);
// 
// XDYNAMICS_API double length(const vector3i &v);
// XDYNAMICS_API double length(const vector3ui &v);
// XDYNAMICS_API double length(const vector3f &v);
// XDYNAMICS_API double length(const vector3d &v);
// 
// XDYNAMICS_API void inverse(matrix33d& A);
// 
// XDYNAMICS_API matrix34d GMatrix(const euler_parameters& e);
// XDYNAMICS_API matrix34d LMatrix(const euler_parameters& e);
// XDYNAMICS_API matrix34d BMatrix(const euler_parameters& e, const vector3d& s);
// XDYNAMICS_API matrix33d GlobalTransformationMatrix(const euler_parameters& e);
// XDYNAMICS_API matrix44d MMatrix(const vector3d& v);
// XDYNAMICS_API matrix44d DMatrix(const vector3d& s, const vector3d& d);
// 
// XDYNAMICS_API int LinearSolve(int n, int nrhs, xMatrixD& a, int lda, xVectorD& b, int ldb);
// 
// // Conversion
// XDYNAMICS_API vector3d ToVector3D(vector3ui& v3);
// XDYNAMICS_API vector3ui ToVector3UI(vector3d& v3);
// 
// #endif