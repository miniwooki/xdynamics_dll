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
	USER_INPUT 
};

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
	XLS_SHAPE = 0,
	XLS_MASS, 
	XLS_JOINT,
	XLS_FORCE,
	XLS_PARTICLE,
	XLS_CONTACT,
	XLS_INTEGRATOR,
	XLS_SIMULATION 
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
	VCUBE, 
	VMESH, 
	VPARTICLE, 
	VTSDA,
	VRAXIAL
};
/*enum xInputDataFormType { FORM_OBJECT_PLANE = 2, FORM_OBJECT_CUBE = 3, FORM_OBJECT_SHAPE = 4 };*/

typedef struct{	double density, youngs, poisson, shear; }xMaterial;
typedef struct{ double dx, dy, drx, dry, drz, pox, poy, poz, p1x, p1y, p1z; }xPlaneObjectData;
typedef struct{ double p0x, p0y, p0z, p1x, p1y, p1z; }xCubeObjectData;
typedef struct{ double dx, dy, dz, lx, ly, lz, minr, maxr; }xCubeParticleData;
typedef struct{ unsigned int number; }xListParticleData;
typedef struct{ double rest, rto, mu, coh; }xContactParameterData;
typedef struct{ double mass, ixx, iyy, izz, ixy, ixz, iyz, px, py, pz, e0, e1, e2, e3, vx, vy, vz; }xPointMassData;
typedef struct{ double lx, ly, lz, fix, fiy, fiz, gix, giy, giz, fjx, fjy, fjz, gjx, gjy, gjz; }xJointData;
typedef struct{ double Ei, Ej, Pri, Prj, Gi, Gj; }xMaterialPair;
typedef struct{ double coh_r, coh_e, kn, vn, ks, vs; }xContactParameters;
typedef struct{ double restitution, friction, rolling_friction, cohesion, stiffness_ratio; }xContactMaterialParameters;
typedef struct{ double spix, spiy, spiz, spjx, spjy, spjz, k, c, init_l; }xTSDAData;

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

inline xContactPairType getContactPair(xShapeType t1, xShapeType t2)
{
	return static_cast<xContactPairType>(t1 + t2);
}

inline xMaterial GetMaterialConstant(int mt)
{
	xMaterial cmt;
	switch (mt){
	case STEEL: cmt.density = STEEL_DENSITY; cmt.youngs = STEEL_YOUNGS_MODULUS; cmt.poisson = STEEL_POISSON_RATIO; cmt.shear = STEEL_SHEAR_MODULUS; break;
	case ACRYLIC: cmt.density = ACRYLIC_DENSITY; cmt.youngs = ACRYLIC_YOUNG_MODULUS; cmt.poisson = ACRYLIC_POISSON_RATIO; cmt.shear = ACRYLIC_SHEAR_MODULUS; break;
	case POLYETHYLENE: cmt.density = POLYETHYLENE_DENSITY; cmt.youngs = POLYETHYLENE_YOUNGS_MODULUS; cmt.poisson = POLYETHYLENE_POISSON_RATIO; cmt.shear = POLYETHYLENE_SHEAR_MODULUS; break;
	case MEDIUM_CLAY: cmt.density = MEDIUM_CLAY_DENSITY; cmt.youngs = MEDIUM_CLAY_YOUNGS_MODULUS; cmt.poisson = MEDIUM_CLAY_POISSON_RATIO; cmt.shear = MEDIUM_SHEAR_MODULUS; break;
	case GLASS: cmt.density = GLASS_DENSITY; cmt.youngs = GLASS_YOUNG_MODULUS; cmt.poisson = GLASS_POISSON_RATIO; cmt.shear = GLASS_SHEAR_MODULUS; break;
	case ALUMINUM: cmt.density = ALUMINUM_DENSITY; cmt.youngs = ALUMINUM_YOUNG_MODULUS; cmt.poisson = ALUMINUM_POISSON_RATIO; cmt.shear = ALUMINUM_SHEAR_MODULUS; break;
	case SAND: cmt.density = SAND_DENSITY; cmt.youngs = SAND_YOUNG_MODULUS; cmt.poisson = SAND_POISSON_RATIO; cmt.shear = SAND_SHEAR_MODULUS; break;
		//case SAND: cmt.density = SAND_DENSITY; cmt.youngs = SAND_YOUNGS_MODULUS; cmt.poisson = SAND_POISSON_RATIO; break;
	}

	return cmt;
}

#endif