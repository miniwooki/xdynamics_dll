#ifndef XRESULTMANAGER_H
#define XRESULTMANAGER_H

#include "xdynamics_decl.h"
#include "xColorControl.h"
#include "xdynamics_object/xPointMass.h"
#include "xmap.hpp"
#include "xlist.hpp"
#include "xdynamics_object/xKinematicConstraint.h"

typedef xPointMass::pointmass_result struct_pmr;
typedef xKinematicConstraint::kinematicConstraint_result struct_kcr;

//class xColorControl;

class XDYNAMICS_API xResultManager
{
public:
	xResultManager();
	~xResultManager();

	void xRun(const std::string _cpath, const std::string _cname);
	void set_num_parts(unsigned int npt);
	unsigned int get_num_parts();
	unsigned int get_num_particles();
	unsigned int get_num_clusters();
	unsigned int get_num_generalized_coordinates();
	unsigned int get_num_constraint_equations();
	
	double* get_times();
	//void set_num_particles(unsigned int np);
	//void set_num_masses_and_joint(unsigned int nm, unsigned int nj);
	//bool initialize();
	float get_min_result_value(xColorControl::ColorMapType cmt);
	float get_max_result_value(xColorControl::ColorMapType cmt);
	struct_pmr* get_mass_result_ptr(std::string n);
	struct_kcr* get_joint_result_ptr(std::string n);
	float* get_particle_position_result_ptr();
	float* get_particle_velocity_result_ptr();
	float* get_particle_color_result_ptr();
	void set_num_generailzed_coordinates(unsigned int ng);
	void set_num_constraints_equations(unsigned int nc);
	bool alloc_time_momory(unsigned int npart);
	bool alloc_dem_result_memory(unsigned int np, unsigned int ns);
	bool alloc_mass_result_memory(std::string name);
	bool alloc_joint_result_memory(std::string name);
	bool save_dem_result(unsigned int i, double* cpos, double* pos, double* vel, double* acc, double* ep, double* ev, double* ea, unsigned int np, unsigned int ns);
	bool save_mass_result(unsigned int i, xPointMass* pm);
	bool save_joint_result(unsigned int i, std::string nm, struct_kcr _kcr);
	bool save_generalized_coordinate_result(double *q, double *qd, double *q_1, double *rhs);
	bool export_step_data_to_file(unsigned int pt, double ct);
	void ExportPointMassResult2TXT(std::string n);
	void setup_particle_buffer_color_distribution(xColorControl* xcc, int sframe, int cframe);

	bool initialize_from_exist_results(std::string path);
	bool upload_exist_results(std::string path);

private:
	void setCurrentPath(std::string new_path);
	void setCurrentName(std::string new_name);

	void ExportBPM2TXT(std::string& file_name);
	void ExportBKC2TXT(std::string& file_name);

	int Execute0(char *d);
	int Execute1(char *d);
	int Execute2(char *d);

	xstring cur_path;// char cur_path[PATH_BUFFER_SIZE];
	xstring cur_name;// char cur_name[NAME_BUFFER_SIZE];

	unsigned int allocated_size;
	unsigned int nparticles;
	unsigned int nclusters;
	unsigned int nparts;
	unsigned int ngeneralized_coordinates;
	unsigned int nconstraints;
	
	xmap<xstring, struct_pmr*> pmrs;
	xmap<xstring, struct_kcr*> kcrs;
	xlist<xstring> flist;
	double* time;
	float* ptrs;
	float* vtrs;
	float* ctrs;

	float max_particle_position[3];
	float min_particle_position[3];
	float max_particle_velocity[3];
	float min_particle_velocity[3];
	float max_particle_position_mag;
	float min_particle_position_mag;
	float max_particle_velocity_mag;
	float min_particle_velocity_mag;

	double *c_cluster_pos;
	double *c_particle_pos;
	double *c_particle_vel;
	double *c_particle_acc;
	double *c_particle_ep;
	double *c_particle_ev;
	double *c_particle_ea;

	double *c_generalized_coord_q;
	double *c_generalized_coord_qd;
	double *c_generalized_coord_q_1;
	double *c_generalized_coord_rhs;

	struct_pmr c_struct_pmr;
	struct_kcr c_struct_kcr;
};

#endif