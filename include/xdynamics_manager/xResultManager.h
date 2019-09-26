#ifndef XRESULTMANAGER_H
#define XRESULTMANAGER_H

#include "xdynamics_decl.h"
#include "xColorControl.h"
#include "xdynamics_object/xPointMass.h"
#include "xmap.hpp"
#include "xlist.hpp"
#include "xdynamics_object/xKinematicConstraint.h"
#include <list>

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
	unsigned int get_terminated_num_parts();
	unsigned int get_current_part_number();
	
	double* get_times();
	//void set_num_particles(unsigned int np);
	//void set_num_masses_and_joint(unsigned int nm, unsigned int nj);
	//bool initialize();
	float get_min_result_value(xColorControl::ColorMapType cmt);
	float get_max_result_value(xColorControl::ColorMapType cmt);
	xlist<xstring>* get_part_file_list();
	struct_pmr* get_mass_result_ptr(std::string n);
	struct_kcr* get_joint_result_ptr(std::string n);
	xmap<xstring, struct_pmr*>* get_mass_result_xmap();
	xmap<xstring, struct_kcr*>* get_joint_result_xmap();
	xlist<unsigned int>* get_distribution_id_list();
	float* get_particle_position_result_ptr();
	float* get_particle_velocity_result_ptr();
	float* get_particle_color_result_ptr();
	void set_gpu_process_device(bool b);
	void set_num_generailzed_coordinates(unsigned int ng);
	void set_num_constraints_equations(unsigned int nc);
	void set_distribution_result(std::list<unsigned int> dl);
	void set_terminated_num_parts(unsigned int _npt);
	void set_p2p_contact_data(int n);
	void set_p2pl_contact_data(int n);
	void set_p2cyl_contact_data(int n);
	void set_p2tri_contact_data(int n);
	bool alloc_time_momory(unsigned int npart);
	bool alloc_dem_result_memory(unsigned int np, unsigned int ns);
	bool alloc_mass_result_memory(std::string name);
	bool alloc_joint_result_memory(std::string name);
	bool alloc_driving_rotation_result_memory(std::string name);
	bool save_dem_result(unsigned int i, double* cpos, double* pos, double* vel, double* acc, double* ep, double* ev, double* ea, unsigned int np, unsigned int ns);
	bool save_mass_result(unsigned int i, xPointMass* pm);
	bool save_joint_result(unsigned int i, std::string nm, struct_kcr _kcr);
	bool save_generalized_coordinate_result(double *q, double *qd, double *q_1, double *rhs);
	bool save_p2p_contact_data(unsigned int* count, unsigned int *id, double* tsd);
	bool save_p2pl_contact_data(unsigned int* count, unsigned int *id, double* tsd);
	bool save_p2cyl_contact_data(unsigned int* count, unsigned int *id, double* tsd);
	bool save_p2tri_contact_data(unsigned int* count, unsigned int *id, double* tsd);
	void save_driving_rotation_result(unsigned int i, std::string nm, unsigned int n_rev, unsigned int dn_rev, double theta);
	bool export_step_data_to_file(unsigned int pt, double ct);
	void export_mass_result_to_text(std::string n);
	void export_joint_result_to_text(std::string n);
	void setup_particle_buffer_color_distribution(xColorControl* xcc, int sframe, int cframe);
	void set_current_part_number(unsigned int cnpt);
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
	bool is_gpu_process;
	unsigned int allocated_size;
	unsigned int nparticles;
	unsigned int nclusters;
	unsigned int nparts;
	unsigned int ncparts;
	unsigned int terminated_num_parts;
	unsigned int ngeneralized_coordinates;
	unsigned int nconstraints;
	int p2p_contact;
	int p2pl_contact;
	int p2cyl_contact;
	int p2tri_contact;
	
	xmap<xstring, struct_pmr*> pmrs;
	xmap<xstring, struct_kcr*> kcrs;
	xmap<xstring, xDrivingRotationResultData> xdrr;
	xlist<xstring> flist;
	xlist<unsigned int> dist_id;
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

	unsigned int *p2p_contact_count;
	unsigned int *p2p_contact_id;
	double *p2p_contact_tsd;

	unsigned int *p2pl_contact_count;
	unsigned int *p2pl_contact_id;
	double *p2pl_contact_tsd;

	unsigned int *p2cyl_contact_count;
	unsigned int *p2cyl_contact_id;
	double *p2cyl_contact_tsd;

	unsigned int *p2tri_contact_count;
	unsigned int *p2tri_contact_id;
	double *p2tri_contact_tsd;

	struct_pmr c_struct_pmr;
	struct_kcr c_struct_kcr;
};

#endif