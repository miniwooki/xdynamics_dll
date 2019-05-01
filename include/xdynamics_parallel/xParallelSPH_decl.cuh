#ifndef XPARALLELSPH_DECL_CUH
#define XPARALLELSPH_DECL_CUH

#include "xdynamics_parallel/xParallelCommon_decl.cuh"
////#include <vector_types.h>
////#include <cuda_runtime.h>
////#include <helper_cuda.h>
////#include "xTypes.h"
//
//struct device_sinusoidal_expression
//{
//	unsigned int sid;
//	unsigned int count;
//	double period;
//	double freq;
//	double c1;
//	double c2;
//	double stime;
//};
//
//struct device_simple_sin_expression
//{
//	unsigned int sid;
//	unsigned int count;
//	double freq;
//	double amp;
//	double stime;
//};
//
//struct device_damping_condition
//{
//	double enable;
//	double alpha;
//	double start_point;
//	double length;
//};
//
//struct device_periodic_condition
//{
//	bool b;
//	double3 dir;
//	double3 limits;
//	double3 velocity;
//};
//
//struct device_pointMass_info
//{
//	double mass;
//	double3 pos;
//	double3 vel;
//	double3 omega;
//	double3 force;
//	double3 moment;
//};
//
//struct device_circle_info
//{
//	unsigned int sid;
//	unsigned int count;
//	double r;
//};
//
void setSPHSymbolicParameter(device_sph_parameters *h_paras);
void XDYNAMICS_API cuBoundaryMoving(unsigned long long int sid, unsigned long long int pcount, double stime, double* pos, double* pos0, double* vel, double* auxVel, unsigned long long int np);
void XDYNAMICS_API cu_sph_calculateHashAndIndex(unsigned int *hashes, unsigned int *cell_id, double *pos, unsigned int np);
void XDYNAMICS_API cu_sph_reorderDataAndFindCellStart(unsigned int *hashes, unsigned int* cell_start, unsigned int np, unsigned int nc);
//void XDYNAMICS_API cuPredict_the_acceleration(double* pos, double* vel, double* acc, double* mass, double* rho, xMaterialType* type, bool* isf, double* corr, double* tbVisc, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuPredict_the_temporal_velocity(double* vel, double* auxVel, double* acc, xMaterialType* type, unsigned int np);
//void XDYNAMICS_API cuCalculation_free_surface(double* pos, double* press, double* mass, double* rho, bool* isf, double* ufs, bool* nearfs, double* d_div_r, xMaterialType* tp, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuCalculation_free_surface_with_shifting(double* pos, double* press, double* mass, bool* isf, double* d_div_r, double* shiftedPos, xMaterialType* tp, unsigned int* hashes, unsigned int* cell_start, unsigned int np);
//void XDYNAMICS_API cuPPE_right_hand_side(double* pos, double* auxVel, double* corr, double* mass, double *rho, bool *fs, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuPressure_poisson_equation(double* pos, double* press, double* corr, double* mass, double* rho, bool* isf, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuUpdate_pressure_residual(double* press, double alpha, double* conj0, double omega, double* conj1, double* tmp1, double* resi, xMaterialType* type, bool* isf, unsigned int np);
//void XDYNAMICS_API cuUpdate_conjugate(double* conj0, double* resi, double beta, double omega, double* tmp0, xMaterialType* type, unsigned int np);
//void XDYNAMICS_API cuUpdate_dummy_pressure_from_boundary(double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int np);
//void XDYNAMICS_API cuCorrect_by_adding_the_pressure_gradient_term(double* pos, double* auxPos, double* vel, double* auxVel, double* acc, double* ufs, double* corr, double* mass, double* rho, double* press, bool* isf, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuKernelCorrection(double* pos, double* corr, double* mass, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, unsigned int np);
//void XDYNAMICS_API cuMixingLengthTurbulence(double *pos, double *vel, double* corr, double *tbVisc, xMaterialType* type, unsigned int* hashes, unsigned int* cell_start, unsigned int np, device_periodic_condition* dpc = NULL);
//void XDYNAMICS_API cuSetViscosityFreeSurfaceParticles(double* pos, double* tbVisc, bool* isf, xMaterialType* type, double* maxVel, unsigned int* hashes, unsigned int* cell_start, unsigned int np);
//void XDYNAMICS_API cuPredict_the_position(double *pos, double *auxPos, double *vel, xMaterialType* type, unsigned int np);
//
//// expression
//void XDYNAMICS_API cuSinusoidalExpression(device_sinusoidal_expression *dse, double* initpos, double* pos, double* vel, double* auxVel, xMaterialType* type, double time, unsigned int np);
////void cuSinusoidalExpressionByData(unsigned int sid, unsigned int count, tExpression* dexps, double* initPos, double* pos, double *vel, double* auxVel, unsigned int step, unsigned int np);
////void cuLinearExpression(unsigned int sid, unsigned int count, double* initPos, double* pos, double *vel, double* auxVel, double time, unsigned int np);
/////void cuSimpleSinExpression(device_simple_sin_expression *dse, double* initpos, double* pos, double* vel, double* auxVel, double time, unsigned int np);
//
//// wave damping
//void XDYNAMICS_API cuWave_damping_formula(device_damping_condition* ddc, double* pos, double* vel, double* auxVel, unsigned int np);
//
//// particle shifting
//void XDYNAMICS_API cuParticleSpacingAverage(double* pos, xMaterialType* type, bool *isf, unsigned int *cell_start, unsigned int *hashes, double* avr, device_periodic_condition* dpc, unsigned int np);
//void XDYNAMICS_API cuParticle_shifting(double* shiftedPos, double* pos, double* shift, double* avr, double* vel, double* mass, double* press, double* rho, xMaterialType *type, double* div_r, bool* isf, device_periodic_condition* dpc, unsigned int *hashes, unsigned int *cell_start, unsigned int np);
//void XDYNAMICS_API cuParticle_shifting_update(double* pos, double* shiftedVel, double* shiftedPress, double* vel, double* press, double* shift, double* mass, double* rho, xMaterialType* type, bool* isf, device_periodic_condition* dpc, unsigned int *hashes, unsigned int *cell_start, unsigned int np);
//
//void XDYNAMICS_API cuReplaceDataByID(double* m_pos, double* m_avel, double* m_mass, double* m_press, double* pos, double* avel, double* mass, double* press, unsigned int* m_id, unsigned int np);
//
////void cuPPE_right_hand_side2(double* pos, double* auxVel, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np);
////void cuPressure_poisson_equation2(double* pos, double* press, double* mass, unsigned int* hashes, unsigned int* cell_start, double* out, unsigned int np);
////
////void cuUpdate_pressure_residual2(double* press, double alpha, double* conj0, double omega, double* conj1, double* tmp1, double* resi, unsigned int np);
////void cuUpdate_conjugate2(double* conj0, double* resi, double beta, double omega, double* tmp0, unsigned int np);
////
////void cuUpdate_dummy_pressure_from_boundary2(double* m_press, double* press, unsigned int* innerDummyNeighbors, xMaterialType* type, bool* isf, unsigned int* m_id, unsigned int np);
//////void cuContactDistance(double* pos, tParticle* type, device_circle_info* dci, unsigned int* hashes, unsigned int* cell_start, unsigned int* cid, double *dist, unsigned int np, unsigned int nc);
////void cuContact_force_circle_boundary(double* pos, xMaterialType* type, device_pointMass_info* dpmi, device_circle_info* dci, device_contact_parameters* dcp, unsigned int* hashes, unsigned int* cell_start, unsigned int np);
//
#endif