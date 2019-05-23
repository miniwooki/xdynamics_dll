#ifndef XTYPES_H
#define XTYPES_H

#define PATH_BUFFER_SIZE 512
#define NAME_BUFFER_SIZE 64

#define STEEL_YOUNGS_MODULUS 2e+011
#define STEEL_DENSITY 7850
#define STEEL_POISSON_RATIO 0.3
#define STEEL_SHEAR_MODULUS 0.0

#define MEDIUM_CLAY_YOUNGS_MODULUS 35E+06
#define MEDIUM_CLAY_DENSITY 1900
#define MEDIUM_CLAY_POISSON_RATIO 0.45
#define MEDIUM_SHEAR_MODULUS 0.0

#define POLYETHYLENE_YOUNGS_MODULUS 1.1E+9
#define POLYETHYLENE_DENSITY		950
#define POLYETHYLENE_POISSON_RATIO	0.42	
#define POLYETHYLENE_SHEAR_MODULUS 0.0

#define GLASS_YOUNG_MODULUS	6.8e+10
#define GLASS_DENSITY		2180
#define GLASS_POISSON_RATIO	0.19
#define GLASS_SHEAR_MODULUS 0.0

#define ACRYLIC_YOUNG_MODULUS 3.2E+009
#define ACRYLIC_DENSITY			1185
#define ACRYLIC_POISSON_RATIO	0.37
#define ACRYLIC_SHEAR_MODULUS 0.0

#define ALUMINUM_YOUNG_MODULUS  70.0E+9
#define ALUMINUM_DENSITY		2700
#define ALUMINUM_POISSON_RATIO	0.34
#define ALUMINUM_SHEAR_MODULUS 0.0

#define SAND_YOUNG_MODULUS 4.0E+7
#define SAND_DENSITY 2600
#define SAND_POISSON_RATIO 0.3
#define SAND_SHEAR_MODULUS 0.0

#define FLUID_YOUNG_MODULUS 0.0
#define FLUID_DENSITY 1000
#define FLUID_POISSON_RATIO 0.0
#define FLUID_SHEAR_MODULUS 0.0

#include <QtCore/QString>

enum xMaterialType
{	
	NO_MATERIAL = 0, 
	STEEL, 
	MEDIUM_CLAY, 
	POLYETHYLENE, 
	GLASS, 
	ACRYLIC, 
	ALUMINUM, 
	SAND, 
	FLUID,
	BOUNDARY,
	DUMMY,
	FLOATING,
	USER_INPUT 
};

inline QString NameOfMaterial(int i)
{
	xMaterialType xmt = (xMaterialType)(i);
	QString n;
	switch (xmt)
	{
	case NO_MATERIAL: n = "No Material"; break;
	case STEEL: n = "Steel"; break;
	case MEDIUM_CLAY: n = "Medium clay"; break;
	case POLYETHYLENE: n = "Polyethylene"; break;
	case GLASS: n = "Glass"; break;
	case ACRYLIC: n = "Acrylic"; break;
	case ALUMINUM: n = "Aluminum"; break;
	case SAND: n = "Sand"; break;
	case USER_INPUT: n = "User input"; break;
	}
	return n;
}

enum xContactPairType
{ 
	NO_CONTACT_PAIR = 0,
	PARTICLE_PARTICLE = 26,
	PARTICLE_CUBE = 15,
	PARTICLE_PANE = 16,
	PARTICLE_MESH_SHAPE = 32,
	PLANE_MESH_SHAPE = 22 
};

enum xShapeType
{ 
	NO_SHAPE = 0,
	CUBE_SHAPE = 2,
	PLANE_SHAPE = 3, 
	LINE_SHAPE = 5, 
	SPHERE_SHAPE = 7,
	MESH_SHAPE = 19, 
	PARTICLES = 13, 
	RECTANGLE_SHAPE = 17,
	CIRCLE_SHAPE = 23, 
	CYLINDER_SHAPE = 27,
	NO_SHAPE_AND_LIST = 99,
	OBJECT = 100 
};
enum xContactForceModelType
{ DHS = 0 };

enum xXlsInputDataType 
{ 
	XLS_KERNEL = 0,
	XLS_SHAPE,
	XLS_MASS, 
	XLS_JOINT,
	XLS_FORCE,	
	XLS_PARTICLE,
	XLS_CONTACT,	
	XLS_INTEGRATOR,
	XLS_SIMULATION,
	XLS_GRAVITY
};

enum xImportShapeType
{ 
	NO_SUPPORT_MESH_SHAPE = 0,
	STL_ASCII 
};

enum xViewObjectType
{ 
	VMARKER = 0,
	VJOINT, 
	VPLANE, 
	VLINE,
	VCUBE, 
	VMESH, 
	VPARTICLE, 
	VTSDA,
	VRAXIAL,
	VSPHPLANE
};

enum xKenelType
{
	CUBIC_SPLINE_KERNEL = 0,
	QUADRATIC_KERNEL,
	QUINTIC_KERNEL,
	WENDLAND_KERNEL,
	GAUSS_KERNEL,
	MODIFIED_GAUSS_KERNEL
};

enum xSPHCorrectionType
{
	NO_CORRECTION = 0,
	GRADIENT_CORRECTION, 
	KERNEL_CORRECTION,
	MIXED_CORRECTION
};

enum xTurbulenceType
{
	NO_TURBULENCE = 0,
	TURBULENCE_MIXING_LENGTH,
	TURBULENCE_K_LM,
	TURBULENCE_K_E
};

enum xBoundaryTreatmentType
{
	DUMMY_PARTICLE_METHOD = 0,
	GHOST_PARTICLE_METHOD
};

enum xUnitType{ MKS = 0, MMKS };
enum xGravityDirection{ PLUS_X = 0, PLUS_Y, PLUS_Z, MINUS_X, MINUS_Y, MINUS_Z };
/*enum xInputDataFormType { FORM_OBJECT_PLANE = 2, FORM_OBJECT_CUBE = 3, FORM_OBJECT_SHAPE = 4 };*/

typedef struct{	double density, youngs, poisson, shear; }xMaterial;
typedef struct{ double p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z; }xPlaneObjectData;
//typedef struct{ double dx, dy, lx, ly, lz; }xSPHPlaneObjectData;
typedef struct{ double p0x, p0y, p0z, p1x, p1y, p1z; }xCubeObjectData;
typedef struct{ double dx, dy, dz, lx, ly, lz, minr, maxr; }xCubeParticleData;
typedef struct{ unsigned int number; }xListParticleData;
typedef struct{ double rest, rto, mu, coh, rf; }xContactParameterData;
typedef struct{ double mass, ixx, iyy, izz, ixy, ixz, iyz, px, py, pz, e0, e1, e2, e3, vx, vy, vz; }xPointMassData;
typedef struct{ double lx, ly, lz, fix, fiy, fiz, gix, giy, giz, fjx, fjy, fjz, gjx, gjy, gjz; }xJointData;
typedef struct{ double Ei, Ej, Pri, Prj, Gi, Gj; }xMaterialPair;
typedef struct{ double coh_r, coh_e, kn, vn, ks, vs; }xContactParameters;
typedef struct{ double restitution, friction, rolling_friction, cohesion, stiffness_ratio, rolling_resistance; }xContactMaterialParameters;
typedef struct{ double spix, spiy, spiz, spjx, spjy, spjz, k, c, init_l; }xTSDAData;
typedef struct{ int correction, dim, type; double factor; }xKernelFunctionData;
typedef struct{ double p0x, p0y, p0z, p1x, p1y, p1z, nx, ny, nz; }xLineObjectData;
typedef struct{ double xx, xy, xz, yy, yz, zz; }symatrix;
typedef struct{ double s0, s1, s2, s3, s4, s5; }double6;
typedef struct{	bool enable; double alpha, start_point, length; }xWaveDampingData;

typedef struct 
{
	double length;
	double r_top, r_bottom;
	double p0x, p0y, p0z;
	double p1x, p1y, p1z;
}xCylinderObjectData;

typedef struct 
{
	double lx, ly, lz;
	double dx, dy, dz;
	double rforce;
}xRotationalAxialForceData;

typedef struct
{ 
	xShapeType type;  
	unsigned int count;
	unsigned int id; 
	double delta_s;
	double dot_s;
	double gab, nx, ny, nz; 
}xPairData;

typedef struct  
{
	double px, py, pz;
	double nx, ny, nz;
	double tx, ty, tz;
}xCorner;

typedef struct
{
	unsigned int sid;
	unsigned int cnt;
	double ivx, ivy, ivz;// iniVel;
	xCorner c1;
	xCorner c2;
	xCorner c3;
	bool inner;
}xOverlapCorner;

inline xContactPairType getContactPair(xShapeType t1, xShapeType t2)
{
	return static_cast<xContactPairType>(t1 + t2);
}

inline xMaterial GetMaterialConstant(int mt)
{
	xMaterial cmt = { 0, };
	switch (mt){
	case STEEL: cmt.density = STEEL_DENSITY; cmt.youngs = STEEL_YOUNGS_MODULUS; cmt.poisson = STEEL_POISSON_RATIO; cmt.shear = STEEL_SHEAR_MODULUS; break;
	case ACRYLIC: cmt.density = ACRYLIC_DENSITY; cmt.youngs = ACRYLIC_YOUNG_MODULUS; cmt.poisson = ACRYLIC_POISSON_RATIO; cmt.shear = ACRYLIC_SHEAR_MODULUS; break;
	case POLYETHYLENE: cmt.density = POLYETHYLENE_DENSITY; cmt.youngs = POLYETHYLENE_YOUNGS_MODULUS; cmt.poisson = POLYETHYLENE_POISSON_RATIO; cmt.shear = POLYETHYLENE_SHEAR_MODULUS; break;
	case MEDIUM_CLAY: cmt.density = MEDIUM_CLAY_DENSITY; cmt.youngs = MEDIUM_CLAY_YOUNGS_MODULUS; cmt.poisson = MEDIUM_CLAY_POISSON_RATIO; cmt.shear = MEDIUM_SHEAR_MODULUS; break;
	case GLASS: cmt.density = GLASS_DENSITY; cmt.youngs = GLASS_YOUNG_MODULUS; cmt.poisson = GLASS_POISSON_RATIO; cmt.shear = GLASS_SHEAR_MODULUS; break;
	case ALUMINUM: cmt.density = ALUMINUM_DENSITY; cmt.youngs = ALUMINUM_YOUNG_MODULUS; cmt.poisson = ALUMINUM_POISSON_RATIO; cmt.shear = ALUMINUM_SHEAR_MODULUS; break;
	case SAND: cmt.density = SAND_DENSITY; cmt.youngs = SAND_YOUNG_MODULUS; cmt.poisson = SAND_POISSON_RATIO; cmt.shear = SAND_SHEAR_MODULUS; break;
	case FLUID: cmt.density = FLUID_DENSITY; cmt.youngs = FLUID_YOUNG_MODULUS; cmt.poisson = FLUID_POISSON_RATIO; break;
	}

	return cmt;
}

#endif