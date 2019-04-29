#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include <helper_math.h>

__device__
uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (dcte.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (dcte.grid_size.y - 1);
	gridPos.z = gridPos.z & (dcte.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, dcte.grid_size.y), dcte.grid_size.x) + __umul24(gridPos.y, dcte.grid_size.x) + gridPos.x;
}

// calculate position in uniform grid
__device__
int3 calcGridPos(double3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - dcte.world_origin.x) / dcte.cell_size);
	gridPos.y = floor((p.y - dcte.world_origin.y) / dcte.cell_size);
	gridPos.z = floor((p.z - dcte.world_origin.z) / dcte.cell_size);
	return gridPos;
}

__global__ void vv_update_position_kernel(double4* pos, double3* vel, double3* acc, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;

	double3 _p = dcte.dt * vel[id] + dcte.half2dt * acc[id];
	pos[id].x += _p.x;
	pos[id].y += _p.y;
	pos[id].z += _p.z;

}

__global__ void vv_update_velocity_kernel(
	double3* vel,
	double3* acc,
	double3* omega,
	double3* alpha,
	double3* force,
	double3* moment,
	double* mass,
	double* iner,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double m = mass[id];
	double3 v = vel[id];
	//double3 L = acc[id];
	double3 av = omega[id];
	//double3 aa = alpha[id];
	double3 a = (1.0 / m) * force[id];
	double3 in = (1.0 / iner[id]) * moment[id];
	v += 0.5 * dcte.dt * (acc[id] + a);
	av += 0.5 * dcte.dt * (alpha[id] + in);// aa;
	//L = (1.0 / m) * force[id];
	//aa = (1.0 / in) * moment[id];
	//v += 0.5 * dcte.dt * L;
	//av += 0.5 * dcte.dt * aa;
	// 	if(id == 0){
	// 		printf("Velocity --- > id = %d -> [%f.6, %f.6, %f.6]\n", id, v.x, v.y, v.z);
	// 	}
	force[id] = m * dcte.gravity;
	moment[id] = make_double3(0.0, 0.0, 0.0);
	vel[id] = v;
	omega[id] = av;
	acc[id] = a;
	alpha[id] = in;
}


__global__ void calculateHashAndIndex_kernel(
	unsigned int* hash, unsigned int* index, double4* pos, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;

	volatile double4 p = pos[id];

	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	/*if(_hash >= dcte.ncell)
	printf("Over limit - hash number : %d", _hash);*/
	hash[id] = _hash;
	index[id] = id;
}

__global__ void calculateHashAndIndexForPolygonSphere_kernel(
	unsigned int* hash, unsigned int* index,
	unsigned int sid, unsigned int nsphere, double4* sphere)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= nsphere) return;
	volatile double4 p = sphere[id];
	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned int _hash = calcGridHash(gridPos);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
}

__global__ void reorderDataAndFindCellStart_kernel(
	unsigned int* hash,
	unsigned int* index,
	unsigned int* cstart,
	unsigned int* cend,
	unsigned int* sorted_index,
	unsigned int np)
{
	extern __shared__ uint sharedHash[];	//blockSize + 1 elements
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned _hash;

	//unsigned int tnp = ;// dcte.np + dcte.nsphere;

	if (id < np)
	{
		_hash = hash[id];

		sharedHash[threadIdx.x + 1] = _hash;

		if (id > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[id - 1];
		}
	}
	__syncthreads();

	if (id < np)
	{
		if (id == 0 || _hash != sharedHash[threadIdx.x])
		{
			cstart[_hash] = id;

			if (id > 0)
				cend[sharedHash[threadIdx.x]] = id;
		}

		if (id == np - 1)
		{
			cend[_hash] = id + 1;
		}

		unsigned int sortedIndex = index[id];
		sorted_index[id] = sortedIndex;
	}
}

__device__ device_force_constant getConstant(
	int tcm, double ir, double jr, double im, double jm,
	double iE, double jE, double ip, double jp,
	double iG, double jG, double rest,
	double fric, double rfric, double sratio)
{
	device_force_constant dfc = { 0, 0, 0, 0, 0, 0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp*jp) + jE*(1 - ip * ip));
	switch (tcm)
	{
	case 0:{
		double Geq = (iG * jG) / (iG*(2 - jp) + jG*(2 - ip));
		double ln_e = log(rest);
		double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.kn * Meq);
		dfc.ks = 8.0 * Geq * sqrt(Req);
		dfc.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.ks * Meq);
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	case 1:{
		double beta = (M_PI / log(rest));
		dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
		dfc.vn = sqrt((4.0 * Meq * dfc.kn) / (1.0 + beta * beta));
		dfc.ks = dfc.kn * sratio;
		dfc.vs = dfc.vn * sratio;
		dfc.mu = fric;
		dfc.ms = rfric;
		break;
	}
	}

	// 	dfu1.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
	// 	dfu1.vn = sqrt((4.0f*em * dfu1.kn) / (1 + beta * beta));
	// 	dfu1.ks = dfu1.kn * ratio;
	// 	dfu1.vs = dfu1.vn * ratio;
	// 	dfu1.mu = fric;
	return dfc;
}

// ref. Three-dimensional discrete element modelling (DEM) of tillage: Accounting for soil cohesion and adhesion
__device__ double cohesionForce(
	double ri,
	double rj,
	double Ei,
	double Ej,
	double pri,
	double prj,
	double coh,
	double Fn)
{
	double cf = 0.f;
	if (coh){
		double req = (ri * rj / (ri + rj));
		double Eeq = ((1.0 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej);
		double rcp = (3.0 * req * (-Fn)) / (4.0 * (1.0 / Eeq));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = coh * Ac;
	}
	return cf;
}

// __device__ bool calForce(
// 	float ir,
// 	float jr,
// 	float im,
// 	float jm,
// 	float rest,
// 	float sh,
// 	float fric,
// 	float rfric,
// 	float E,
// 	float pr,
// 	float coh,
// 	float4 ipos,
// 	float4 jpos,
// 	float3 ivel,
// 	float3 jvel,
// 	float3 iomega,
// 	float3 jomega,
// 	float3& force,
// 	float3& moment
// 	/*float *riv*/)
// {
// 	float3 relative_pos = make_float3(jpos - ipos);
// 	float dist = length(relative_pos);
// 	float collid_dist = (ir + jr) - dist;
// 	float3 shear_force = make_float3(0.f);
// 	if (collid_dist <= 0){
// 		//*riv = 0.f;
// 		return false;
// 	}
// 	else{
// 		float rcon = ir - 0.5f * collid_dist;
// 		float3 unit = relative_pos / dist;
// 		float3 relative_vel = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
// 		//*riv = abs(length(relative_vel));
// 		device_force_constant<float> c = getConstant<float>(ir, jr, im, jm, E, E, pr, pr, sh, sh, rest, fric, rfric);
// 		float fsn = -u1.kn * pow(collid_dist, 1.5f);
// 		float fca = cohesionForce(ir, jr, E, E, pr, pr, coh, fsn);
// 		float fsd = u1.vn * dot(relative_vel, unit);
// 		float3 single_force = (fsn + fca + fsd) * unit;
// 		//float3 single_force = (-u1.kn * pow(collid_dist, 1.5f) + u1.vn * dot(relative_vel, unit)) * unit;
// 		float3 single_moment = make_float3(0, 0, 0);
// 		float3 e = relative_vel - dot(relative_vel, unit) * unit;
// 		float mag_e = length(e);
// 		if (mag_e){
// 			float3 s_hat = e / mag_e;
// 			float ds = mag_e * dcte.dt;
// 			float fst = -u1.ks * ds;
// 			float fdt = u1.vs * dot(relative_vel, s_hat);
// 			shear_force = (fst + fdt) * s_hat;
// 			if (length(shear_force) >= u1.mu * length(single_force))
// 				shear_force = u1.mu * fsn * s_hat;
// 			single_moment = cross(rcon * unit, shear_force);
// 			if (length(iomega)){
// 				float3 on = iomega / length(iomega);
// 				single_moment += -rfric * fsn * rcon * on;
// 			}
// 			//shear_force = min(u1.ks * ds + u1.vs * (dot(relative_vel, s_hat)), u1.mu * length(single_force)) * s_hat;
// 			//single_moment = cross(ir * unit, shear_force);
// 		}
// 		force += single_force + shear_force;
// 		moment += single_moment;
// 	}
// 	
// 	return true;
// }
__device__ void HMCModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
	// 	if (coh && cdist < 1.0E-8)
	// 		return;

	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	// 	if ((fsn + fca + fdn) < 0 && ir)
	// 		return;
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e){
		double3 s_hat = -(e / mag_e);
		double ds = mag_e * dcte.dt;
		double fst = -c.ks * ds;
		double fdt = c.vs * dot(dv, s_hat);
		Ft = (fst + fdt) * s_hat;
		if (length(Ft) >= c.mu * length(Fn))
			Ft = c.mu * fsn * s_hat;
		M = cross(ir * unit, Ft);
		if (length(iomega)){
			double3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}
	}
}

__device__ void DHSModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double rcon, double cdist, double3 iomega, double& _ds, double& dots,
	double3 dv, double3 unit, double3& Ft, double3& Fn, double3& M)
{
	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e)
	{
		double3 sh = e / mag_e;
		double s_dot = dot(dv, sh);
		double ds = _ds + dcte.dt * (s_dot + dots);
		_ds = ds;
		dots = s_dot;
		//double ds = mag_e * dcte.dt;
		double ft0 = c.ks * ds + c.vs * (dot(dv, sh));
		double ft1 = c.mu * length(Fn);
		Ft = min(ft0, ft1) * sh;
		M = cross(ir * unit, Ft);
		/*if (length(iomega)){
		double3 on = iomega / length(iomega);
		M += u1.ms * fsn * rcon * on;
		}*/
	}
}

//__global__ void check_no_collision_pair_kernel(double4* pos, uint2* pinfo, unsigned int* pother, unsigned int _np)
//{
//	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	unsigned int np = _np;
//	if (id >= np)
//		return;
//	uint2 pid = pinfo[id];
//	if (!pid.y)
//		return;
//	unsigned int sid = id * MAXIMUM_PAIR_NUMBER;
//	unsigned int count = 0;
//	double4 ipos = pos[pid.x];
//	unsigned int other = 0;
//	while (count < MAXIMUM_PAIR_NUMBER && other != 0xffffffff)
//	{
//		other = pother[sid + count];
//		double4 jpos = pos[other];
//		double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
//		double dist = length(rp);
//		double cdist = (ipos.w + jpos.w) - dist;
//		if (cdist <= 0)
//		{
//			pother[sid + count] = 0xffffffff;
//			//data.y = 0xffffffff;
//			pid.y--;
//		}
//		count++;
//	}
//	pinfo[id] = pid;
//}
//
//__global__ void check_new_collision_pair_kernel(
//	double4* pos,  uint2 *pinfo, unsigned int* pdata,
//	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int _np)
//{
//	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	unsigned int np = _np;
//	if (id >= np)
//		return;
//	double4 ipos = pos[id];
//	double4 jpos = make_double4(0, 0, 0, 0);
//	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
//
//	double ir = ipos.w; double jr = 0;
//	int3 neighbour_pos = make_int3(0, 0, 0);
//	uint grid_hash = 0;
//	unsigned int start_index = 0;
//	unsigned int end_index = 0;
//	uint2 pid = pinfo[id];
//	unsigned int sid = id * MAXIMUM_PAIR_NUMBER;
//	for (int z = -1; z <= 1; z++){
//		for (int y = -1; y <= 1; y++){
//			for (int x = -1; x <= 1; x++){
//				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
//				grid_hash = calcGridHash(neighbour_pos);
//				start_index = cstart[grid_hash];
//				if (start_index != 0xffffffff){
//					end_index = cend[grid_hash];
//					for (unsigned int j = start_index; j < end_index; j++){
//						unsigned int k = sorted_index[j];
//						if (id == k || k >= np)
//							continue;
//						jpos = pos[k];
//						jr = jpos.w;
//						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
//						double dist = length(rp);
//						double cdist = (ir + jr) - dist;
//						if (cdist > 0){
//							pid.y++;
//							for (unsigned int i = sid; i < sid + MAXIMUM_PAIR_NUMBER; i++)
//							{
//								if (pdata[sid] != 0xffffffff)
//								{
//									pdata[sid] = k;
//									break;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	pinfo[id] = pid;
//}
//
//__global__ void calculate_particle_collision_with_pair_kernel(
//	double4* pos, double3* vel, double3* omega,
//	double* mass, double2* tan, double3* force, double3* moment, 
//	uint2* pinfo, unsigned int* pother, device_contact_property* cp, unsigned int np)
//{
//	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	if (id >= np)
//		return;
//	uint2 pid = pinfo[id];
//	if (!pid.y)
//		return;
//	double im = mass[pid.x];
//	double jm = 0.0;
//	double4 ipos = pos[pid.x];
//	double3 ivel = vel[pid.x];
//	double3 iomega = omega[pid.x];
//	double4 jpos = make_double4(0.0, 0.0, 0.0, 0.0);
//	double3 jvel = make_double3(0.0, 0.0, 0.0);
//	double3 jomega = make_double3(0.0, 0.0, 0.0);
//	double3 Ft = make_double3(0, 0, 0);
//	double3 Fn = make_double3(0, 0, 0);// [id] * dcte.gravity;
//	double3 M = make_double3(0, 0, 0);
//	double3 sumF = make_double3(0, 0, 0);
//	double3 sumM = make_double3(0, 0, 0);
//	unsigned int sid = id * MAXIMUM_PAIR_NUMBER;
//	for (unsigned int i = 0; i < MAXIMUM_PAIR_NUMBER; i++)
//	{
//		unsigned int other = pother[sid + i];
//		if (other == 0xffffffff)
//			continue;
//		jpos = pos[other];
//		jvel = vel[other];
//		jomega = omega[other];
//		jm = mass[other];
//		double2 ds = tan[sid + i];
//		double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
//		double dist = length(rp);
//		double cdist = (ipos.w + jpos.w) - dist;
//		double rcon = ipos.w - 0.5 * cdist;
//		double3 unit = rp / dist;
//		double3 rv = jvel + cross(jomega, -jpos.w * unit) - (ivel + cross(iomega, ipos.w * unit));
//		device_force_constant c = getConstant(
//			1, ipos.w, jpos.w, im, jm, cp->Ei, cp->Ej,
//			cp->pri, cp->prj, cp->Gi, cp->Gj,
//			cp->rest, cp->fric, cp->rfric, cp->sratio);
//		DHSModel(
//			c, ipos.w, jpos.w, cp->Ei, cp->Ej, cp->pri, cp->prj,
//			cp->coh, rcon, cdist, iomega, ds.x, ds.y,
//			rv, unit, Ft, Fn, M);
//		break;
//
//		sumF += Fn + Ft;
//		sumM += M;
//	}
//	force[pid.x] += sumF;
//	moment[pid.x] += sumM;
//}

/*__global__ void calculate_pair_count_kernel(
	double4* pos, uint2* pidx, uint2* pdata,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, unsigned int np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	double ir = ipos.w;
	double jr = 0.0;
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k];
						jr = jpos.w;
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0)
						{
							count[id]++;
						}
					}
				}
			}
		}
	}
}*/

template <int TCM>
__global__ void calculate_p2p_kernel(
	double4* pos, double3* vel,
	double3* omega, double3* force,
	double3* moment, double* mass,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np)
{
	//unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//if (id >= np)
	//	return;

	//double4 ipos = pos[id];
	//double4 jpos = make_double4(0, 0, 0, 0);
	//double3 ivel = vel[id];
	//double3 jvel = make_double3(0, 0, 0);
	//double3 iomega = omega[id];
	//double3 jomega = make_double3(0, 0, 0);
	//int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));

	//double ir = ipos.w; double jr = 0;
	//double im = mass[id]; double jm = 0;
	//double3 Ft = make_double3(0, 0, 0);
	//double3 Fn = make_double3(0, 0, 0);// [id] * dcte.gravity;
	//double3 M = make_double3(0, 0, 0);
	//double3 sumF = make_double3(0, 0, 0);
	//double3 sumM = make_double3(0, 0, 0);
	//int3 neighbour_pos = make_int3(0, 0, 0);
	//uint grid_hash = 0;
	//unsigned int start_index = 0;
	//unsigned int end_index = 0;
	//for (int z = -1; z <= 1; z++){
	//	for (int y = -1; y <= 1; y++){
	//		for (int x = -1; x <= 1; x++){
	//			neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
	//			grid_hash = calcGridHash(neighbour_pos);
	//			start_index = cstart[grid_hash];
	//			if (start_index != 0xffffffff){
	//				end_index = cend[grid_hash];
	//				for (unsigned int j = start_index; j < end_index; j++){
	//					unsigned int k = sorted_index[j];
	//					if (id == k || k >= np)
	//						continue;
	//					jpos = pos[k]; jvel = vel[k]; jomega = omega[k];
	//					jr = jpos.w; jm = mass[k];
	//					double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
	//					double dist = length(rp);
	//					double cdist = (ir + jr) - dist;
	//					if (cdist > 0){
	//						double rcon = ir - 0.5 * cdist;
	//						double3 unit = rp / dist;
	//						double3 rv = dcte.rollingCondition ? jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit)) : (jvel - ivel);
	//						device_force_constant c = getConstant(
	//							TCM, ir, jr, im, jm, cp->Ei, cp->Ej,
	//							cp->pri, cp->prj, cp->Gi, cp->Gj,
	//							cp->rest, cp->fric, cp->rfric, cp->sratio);
	//					/*	switch (TCM)
	//						{
	//						case 0:
	//							HMCModel(
	//								c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
	//								cp->coh, rcon, cdist, iomega,
	//								rv, unit, Ft, Fn, M);
	//							break;
	//						case 1:
	//							DHSModel(
	//								c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
	//								cp->coh, rcon, cdist, iomega,
	//								rv, unit, Ft, Fn, M);
	//							break;
	//						}*/
	//						sumF += Fn + Ft;
	//						sumM += M;
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	//force[id] += sumF;
	//moment[id] += sumM;
}

__device__ double particle_plane_contact_detection(
	device_plane_info& pe, double3& xp, double3& wp, double3& u, double r)
{
	double a_l1 = pow(wp.x - pe.l1, 2.0);
	double b_l2 = pow(wp.y - pe.l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r*r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe.l1) && (wp.y > 0 && wp.y < pe.l2)){
		double3 dp = xp - pe.xw;
		double3 uu = pe.uw / length(pe.uw);
		int pp = -sign(dot(dp, pe.uw));// dp.dot(pe.UW()));
		u = pp * uu;
		double collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr){
		double3 Xsw = xp - pe.w2;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y > pe.l2 && (a_l1 + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w3;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe.l2 && (sqa + b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w4;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe.l1) && wp.y < 0 && (sqb + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w2 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe.l1) && wp.y > pe.l2 && (b_l2 + sqc) < sqr){
		double3 Xsw = xp - pe.w4;
		double3 wj_wi = pe.w3 - pe.w4;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x < 0 && (sqr + sqc) < sqr){
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w4 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x > pe.l1 && (a_l1 + sqc) < sqr){
		double3 Xsw = xp - pe.w2;
		double3 wj_wi = pe.w3 - pe.w2;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	return -1.0;
}

__device__ double3 closestPtPointTriangle(
	device_triangle_info& dpi,
	double3& p,
	double pr,
	int& ct)
{
	double3 a = dpi.P;
	double3 b = dpi.Q;
	double3 c = dpi.R;
	double3 ab = b - a;
	double3 ac = c - a;
	double3 ap = p - a;
	double3 bp = p - b;
	double3 cp = p - c;
	double d1 = dot(ab, ap);
	double d2 = dot(ac, ap);
	double d3 = dot(ab, bp);
	double d4 = dot(ac, bp);
	double d5 = dot(ab, cp);
	double d6 = dot(ac, cp);
	double va = d3 * d6 - d5 * d4;
	double vb = d5 * d2 - d1 * d6;
	double vc = d1 * d4 - d3 * d2;

	
	if (d1 <= 0.0 && d2 <= 0.0){ ct = 0; return a; }
	if (d3 >= 0.0 && d4 <= d3) { ct = 0; return b; }
	if (d6 >= 0.0 && d5 <= d6) { ct = 0; return c; }

	if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){ ct = 1; return a + (d1 / (d1 - d3)) * ab; }
	if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){ ct = 1; return a + (d2 / (d2 - d6)) * ac; }
	if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){ ct = 1;	return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b); }
	//ct = 2;
	// P inside face region. Comu0te Q through its barycentric coordinates (u, v, w)
	/*double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;*/
	double denom = 1.0 / (va + vb + vc);
	double3 v = vb * denom * ab;
	double3 w = vc * denom * ac;
	double3 _cpt = a + v + w;
	//double _dist = pr - length(p - _cpt);
	ct = 2;
	//if (_dist > 0) return _cpt;
	return _cpt; // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
}

__global__ void calculate_particle_particle_contact_count(
	double4* pos, pair_data* old_pppd, unsigned int *old_count, unsigned int *count,
	unsigned int* sidx, unsigned int *sorted_index, unsigned int *cstart,
	unsigned int* cend, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int old_sid = 0;
	if (id != 0)
		old_sid = sidx[id-1];
	unsigned int cnt = 0;
	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = ipos.w; double jr = 0;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)	continue;
						jpos = pos[k];
						jr = jpos.w;
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0) cnt++;
						else{
							for (unsigned int j = 0; j < old_count[id]; j++){
								pair_data *pdata = old_pppd + (old_sid + j);
								if (pdata->type == 1) pdata->enable = false;
							}
						}
					}
				}
			}
		}
	}
	//old_count[id] = count[id];
	count[id] = cnt;
}

__device__ bool checkOverlab(int3 ctype, double3 p, double3 c, double3 u0, double3 u1)
{
	bool b_over = false;
	if (p.x >= u1.x - 1e-9 && p.x <= u1.x + 1e-9)
		if (p.y >= u1.y - 1e-9 && p.y <= u1.y + 1e-9)
			if (p.z >= u1.z - 1e-9 && p.z <= u1.z + 1e-9)
				b_over = true;

	if (/*(ctype.y || ctype.z) &&*/ !b_over)
	{
		if (u0.x >= u1.x - 1e-9 && u0.x <= u1.x + 1e-9)
			if (u0.y >= u1.y - 1e-9 && u0.y <= u1.y + 1e-9)
				if (u0.z >= u1.z - 1e-9 && u0.z <= u1.z + 1e-9)
					b_over = true;
	}
	return b_over;
}

__global__ void calculate_particle_triangle_contact_count(
	device_triangle_info* dpi, double4 *pos, pair_data* old_pppd, 
	unsigned int* old_count, unsigned int* count,
	unsigned int* sidx,	unsigned int* sorted_index, 
	unsigned int* cstart, unsigned int* cend, unsigned int _np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np) return;
	double cdist = 0.0;
	double4 ipos4 = pos[id];
	double3 ipos = make_double3(ipos4.x, ipos4.y, ipos4.z);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPos(ipos);
	double ir = pos[id].w;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int cnt = 0;
	unsigned int old_sid = 0;
//	unsigned int overlap_count = 0;
	int3 ctype = make_int3(0, 0, 0);
	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	if (id != 0) old_sid = sidx[id - 1];
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= np){
							k -= np;
							int t = -1;
//							unsigned int pidx = dpi[k].id;
							device_triangle_info tri = dpi[k];
							double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir, t);
							double3 unit = -normalize(cross(tri.Q - tri.P, tri.R - tri.P));
							//double angle = dot(pc_unit, unit) / (length(pc_unit) * length(unit));*/
							cdist = ir - length(ipos - cpt);
							if (cdist > 0)
							{
								//double len_cpt = sqrt(dot(cpt, cpt));
							//	cpt = len_cpt ? cpt / sqrt(dot(cpt, cpt)) : cpt;								
								bool overlap = checkOverlab(ctype, previous_cpt, cpt, previous_unit, unit);
								if (overlap)
									continue;
								previous_cpt = cpt;
								previous_unit = unit;
								*(&(ctype.x) + t) += 1;
								cnt++;
							}								
							else{
								for (unsigned int j = 0; j < old_count[id]; j++){
									pair_data *pdata = old_pppd + (old_sid + j);
									if (pdata->type == 2) pdata->enable = false;
								}
							}
						}
					}
				}
			}
		}
	}
	if (cnt > 1)
		cnt = cnt;
	count[id] += cnt;
}

__global__ void calculate_particle_plane_contact_count(
	device_plane_info *plane, pair_data* old_pppd, unsigned int *old_count, unsigned int *count,
	unsigned int *sidx, double4* pos, unsigned int _nplanes, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	unsigned int nplanes = _nplanes;
	if (id >= np)
		return;
	//count[id] = 0.0;
	double4 ipos;
	ipos.x = pos[id].x;
	ipos.y = pos[id].y;
	ipos.z = pos[id].z;
	ipos.w = pos[id].w;// (pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	unsigned int old_sid = 0;
	if(id != 0)
		old_sid = sidx[id-1];
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < nplanes; i++)
	{
		device_plane_info pl = plane[i];
		double3 unit = make_double3(0, 0, 0);
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));
		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, ipos.w);
		if (cdist > 0)
			cnt++;
		else
		{
			for (unsigned int j = 0; j < old_count[id]; j++)
			{
				pair_data *pdata = old_pppd + (old_sid + j);
				if (pdata->j == i && pdata->type == 0)
					pdata->enable = false;
			}
		}
	}
	//old_count[id] = count[id];
	count[id] += cnt;
}

__global__ void copy_old_to_new_pair(
	unsigned int *old_count, unsigned int *new_count, unsigned int* old_sidx, unsigned int* new_sidx,
	pair_data* old_pppd, pair_data* new_pppd, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.y) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int oc = old_count[id];
//	unsigned int nc = new_count[id];
	//particle_plane_pair_data new_ppp[10] = { 0, };
	unsigned int old_sid = 0;
	unsigned int new_sid = 0;
	if (id != 0)
	{
		old_sid = old_sidx[id - 1];
		new_sid = new_sidx[id - 1];
	}	
	unsigned int new_cnt = 0;
	for (unsigned int i = 0; i < oc; i++)
	{
		if (old_pppd[old_sid + i].enable)
		{
			new_pppd[new_sid + new_cnt++] = old_pppd[old_sid + i];
		}
	}
}

__global__ void new_particle_particle_contact_kernel(
	double4* pos, double3* vel, double3* omega,
	double* mass, double3* force, double3* moment, pair_data *old_pairs, pair_data *pairs,
	unsigned int* old_count, unsigned int* count, unsigned int* old_sidx, unsigned int* sidx, int2* type_count, device_contact_property* cp,
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np)
		return;
	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	double3 ivel = vel[id];
	double3 iomega = omega[id];
	double3 jvel = make_double3(0.0, 0.0, 0.0);
	double3 jomega = make_double3(0.0, 0.0, 0.0);
	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	double3 Fn = make_double3(0, 0, 0);// [id] * dcte.gravity;
	double3 M = make_double3(0, 0, 0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = ipos.w; double jr = 0;
	double im = mass[id]; double jm = 0.0;
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int old_sid = 0;
	unsigned int sid = 0;
	unsigned int old_cnt = old_count[id];
//	unsigned int cnt = count[id];
	pair_data pair;
	int tcnt = 0;
	if (id != 0)
	{
		sid = sidx[id - 1];
		old_sid = old_sidx[id - 1];
	}
		
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k];
						jr = jpos.w;
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0)
						{
							pair = { true, 1, id, k, 0.0, 0.0 };
							for (unsigned int j = 0; j < old_cnt; j++)
							{
								pair_data* pd = old_pairs + (old_sid + j);
								if (pd->enable && pd->j == k)
								{
									pair = *pd;
									break;
								}
							}
							jvel = vel[k];
							jomega = omega[k];
							jm = mass[k];
							double rcon = ipos.w - 0.5 * cdist;
							double3 unit = rp / dist;
							double3 rv = jvel + cross(jomega, -jpos.w * unit) - (ivel + cross(iomega, ipos.w * unit));
							device_force_constant c = getConstant(
								1, ipos.w, jpos.w, im, jm, cp->Ei, cp->Ej,
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->rfric, cp->sratio);
							DHSModel(
								c, ipos.w, jpos.w, cp->Ei, cp->Ej, cp->pri, cp->prj,
								cp->coh, rcon, cdist, iomega, pair.ds, pair.dots,
								rv, unit, Ft, Fn, M);
							sumF += Fn + Ft;
							sumM += M;
							pairs[sid + tcnt++] = pair;
						}
					}
				}
			}
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
	type_count[id].x = tcnt;
}

__global__ void new_particle_polygon_object_conatct_kernel(
	device_triangle_info* dpi, device_mesh_mass_info* dpmi, pair_data* old_pairs, pair_data* pairs,
	unsigned int* old_count, unsigned int* count, unsigned int* old_sidx, unsigned int* sidx, int2* type_count,
	double4 *pos, double3 *vel, double3 *omega, double3 *force, double3 *moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property *cp, unsigned int _np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	if (id >= np) return;
	unsigned int old_sid = 0;
	unsigned int sid = type_count[id].x + type_count[id].y;
	if (id != 0){
		old_sid = old_sidx[id - 1];
		sid += sidx[id - 1];
	}
	double cdist = 0.0;
	double im = mass[id];
	double4 ipos4 = pos[id];
	double3 ipos = make_double3(ipos4.x, ipos4.y, ipos4.z);
	double3 ivel = vel[id];
	double3 iomega = omega[id];
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = ipos4.w;
	double3 M = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 sum_force = make_double3(0, 0, 0);
	double3 sum_moment = make_double3(0, 0, 0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int old_cnt = old_count[id];
//	unsigned int cnt = count[id];
	pair_data pair;
	int3 ctype = make_int3(0, 0, 0);
	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	for (int z = -1; z <= 1; z++){
		for (int y = -1; y <= 1; y++){
			for (int x = -1; x <= 1; x++){
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff){
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++){
						unsigned int k = sorted_index[j];
						if (k >= dcte.np)
						{
							int t = -1;
							k -= dcte.np;
							unsigned int pidx = dpi[k].id;
							device_contact_property cmp = cp[pidx];
							device_mesh_mass_info pmi = dpmi[pidx];
							double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir, t);
							double3 po2cp = cpt - pmi.origin;
							cdist = ir - length(ipos - cpt);
							Fn = make_double3(0.0, 0.0, 0.0);
							if (cdist > 0)
							{
								//double len_cpt = sqrt(dot(cpt, cpt));
								//cpt = len_cpt ? cpt / sqrt(dot(cpt, cpt)) : cpt;
								device_triangle_info tri = dpi[k];
								double3 qp = tri.Q - tri.P;
								double3 rp = tri.R - tri.P;
								double rcon = ir - 0.5 * cdist;
								unit = -normalize(cross(qp, rp));// unit / length(cross(qp, rp));
								bool overlab = checkOverlab(ctype, previous_cpt, cpt, previous_unit, unit);
								if (overlab)
									continue;
								*(&(ctype.x) + t) += 1;
								//previous_cpt = cpt;
								pair = { true, 2, id, k, 0.0, 0.0 };
								for (unsigned int j = 0; j < old_cnt; j++)
								{
									pair_data* pd = old_pairs + old_sid + j;
									if (pd->enable && pd->j == k)
									{
										pair = *pd;
										break;
									}
								}								
								previous_cpt = cpt;
								previous_unit = unit;
								*(&(ctype.x) + t) += 1;
								double3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, ir * unit));
								device_force_constant c = getConstant(
									1, ir, 0, im, 0, cmp.Ei, cmp.Ej,
									cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
									cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
								DHSModel(
									c, ir, 0, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega, pair.ds, pair.dots,
									dv, unit, Ft, Fn, M);
								sum_force += Fn + Ft;
								sum_moment += M;
								dpmi[pidx].force += -(Fn + Ft);
								dpmi[pidx].moment += -cross(po2cp, Fn + Ft);
							}
						}
					}
				}
			}
		}
	}
	force[id] += sum_force;
	moment[id] += sum_moment;
}

__global__ void new_particle_plane_contact(
	device_plane_info *plane, double4* pos, double3* vel, 
	double3* omega, double* mass, double3* force, 
	double3* moment, unsigned int* old_count, unsigned int *count, 
	unsigned int* old_sidx, unsigned int *sidx, int2 *type_count,
	pair_data *old_pairs, pair_data *pairs, device_contact_property *cp, 
	unsigned int _nplanes, unsigned int _np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = _np;
	unsigned int nplanes = _nplanes;
	if (id >= np)
		return;
	unsigned int old_sid = 0;
	unsigned int sid = type_count[id].x;
	if (id != 0)
	{
		old_sid = old_sidx[id - 1];
		sid = sidx[id - 1] + type_count[id].x;
	}
		
	double4 ipos = pos[id];
	double3 ivel = vel[id];
	double3 iomega = omega[id];
	double m = mass[id];
	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	unsigned int old_cnt = old_count[id];
//	unsigned int cnt = count[id];
	pair_data ppp;
	unsigned int cur_cnt = 0;
	for (unsigned int i = 0; i < nplanes; i++)
	{
		ppp = { false, 0, id, i + np, 0.0, 0.0 };
		for (unsigned int j = 0; j < old_cnt; j++)
		{
			pair_data* pair = old_pairs + old_sid + j;
			if (pair->enable && pair->j == i + np)
			{
				ppp = *pair;// pairs[sid + j]
				break;
			}
		}
		
		device_plane_info pl = plane[ppp.j - np];
		double3 unit = make_double3(0, 0, 0);
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));
		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, ipos.w);
		if (cdist > 0)
		{
			ppp.enable = true;
			ppp.type = 2;
			double rcon = ipos.w - 0.5 * cdist;
			double3 dv = -(ivel + cross(iomega, ipos.w * unit));
			device_force_constant c = getConstant(
				1, ipos.w, 0.0, m, 0.0, cp->Ei, cp->Ej,
				cp->pri, cp->prj, cp->Gi, cp->Gj,
				cp->rest, cp->fric, cp->rfric, cp->sratio);
			DHSModel(
				c, ipos.w, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
				iomega, ppp.ds, ppp.dots, dv, unit, Ft, Fn, M);
			force[id] += Fn + Ft;// m_force + shear_force;
			moment[id] += M;
			pairs[sid + cur_cnt++] = ppp;
		}
	}
	type_count[id].y = cur_cnt;
}

template <int TCM>
__global__ void plane_contact_force_kernel(
	device_plane_info *plane,
	double4* pos, double3* vel, double3* omega,
	double3* force, double3* moment,
	device_contact_property *cp, double* mass, unsigned int np)
{
	//unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//if (id >= np)
	//	return;
	//double m = mass[id];
	//double4 ipos = pos[id];
	//double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	//double r = ipos.w;
	//double3 ivel = vel[id];
	//double3 iomega = omega[id];

	//double3 Fn = make_double3(0, 0, 0);
	//double3 Ft = make_double3(0, 0, 0);
	//double3 M = make_double3(0, 0, 0);
	//double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - plane->xw;
	//double3 unit = make_double3(0, 0, 0);
	//double3 wp = make_double3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));

	//double cdist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
	//if (cdist > 0){
	//	double rcon = r - 0.5 * cdist;
	//	double3 dv = dcte.rollingCondition ? -(ivel + cross(iomega, r * unit)) : -ivel;
	//	device_force_constant c = getConstant(
	//		TCM, r, 0.0, m, 0.0, cp->Ei, cp->Ej,
	//		cp->pri, cp->prj, cp->Gi, cp->Gj,
	//		cp->rest, cp->fric, cp->rfric, cp->sratio);
	//	switch (TCM)
	//	{
	//	case 0:
	//		HMCModel(
	//			c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
	//			iomega, dv, unit, Ft, Fn, M);
	//		break;
	//	case 1:
	//		DHSModel(
	//			c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
	//			iomega, dv, unit, Ft, Fn, M);
	//		break;
	//	}
	//}
	//force[id] += Fn + Ft;// m_force + shear_force;
	//moment[id] += M;
}


__device__ double3 makeTFM_1(double4& ep)
{
	return make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
}

__device__ double3 makeTFM_2(double4& ep)
{
	return make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
}

__device__ double3 makeTFM_3(double4& ep)
{
	return make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
}

__device__ double3 toLocal(double3& A_1, double3& A_2, double3& A_3, double3& v)
{
	return make_double3
		(A_1.x * v.x + A_2.x * v.y + A_3.x * v.z,
		A_1.y * v.x + A_2.y * v.y + A_3.y * v.z,
		A_1.z * v.x + A_2.z * v.y + A_3.z * v.z);
}

__device__ double3 toGlobal(double3& A_1, double3& A_2, double3& A_3, double3& v)
{
	return make_double3(dot(A_1, v), dot(A_2, v), dot(A_3, v));
}

__device__ float particle_cylinder_contact_detection(
	device_cylinder_info* cy, double4& pt, double3& u, double3& cp, unsigned int id = 0)
{
	double dist = -1.0;
	double3 ab = make_double3(cy->ptop.x - cy->pbase.x, cy->ptop.y - cy->pbase.y, cy->ptop.z - cy->pbase.z);
	double3 p = make_double3(pt.x, pt.y, pt.z);
	double t = dot(p - cy->pbase, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	if (t >= 0 && t <= 1){
		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		u = (_cp - p) / dist;
		cp = _cp - cy->rbase * u;
		return cy->rtop + pt.w - dist;
	}
	else{

		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		if (dist < cy->rbase){
			double3 OtoCp = cy->origin - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - cy->rbase * u;
			return cy->len * 0.5 + pt.w - OtoCp_;
		}
		double3 A_1 = makeTFM_1(cy->ep);
		double3 A_2 = makeTFM_2(cy->ep);
		double3 A_3 = makeTFM_3(cy->ep);
		double3 _at = p - cy->ptop;
		double3 at = toLocal(A_1, A_2, A_3, _at);
		double r = length(at);
		cp = cy->ptop;
		if (abs(at.y) > cy->len){
			_at = p - cy->pbase;
			at = toLocal(A_1, A_2, A_3, _at);
			cp = cy->pbase;
		}
		double pi = atan(at.x / at.z);
		if (pi < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else if (pi > 0 && at.x < 0 && at.z < 0){
			_cp.x = cy->rbase * sin(-pi);
		}
		else{
			_cp.x = cy->rbase * sin(pi);
		}
		_cp.z = cy->rbase * cos(pi);
		if (at.z < 0 && _cp.z > 0){
			_cp.z = -_cp.z;
		}
		else if (at.z > 0 && _cp.z < 0){
			_cp.z = -_cp.z;
		}
		_cp.y = 0.;
		cp = cp + toGlobal(A_1, A_2, A_3, _cp);

		double3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < pt.w){
			return pt.w - dist;
		}
	}
	return -1.0;
}

template<int TCM>
__global__ void cylinder_hertzian_contact_force_kernel(
	device_cylinder_info *cy,
	double4* pos, double3* vel, double3* omega,
	double3* force, double3* moment, device_contact_property *cp,
	double* mass, double3* mpos, double3* mf, double3* mm, unsigned int np)
{
	//unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//if (id >= np)
	//	return;

	//*mf = make_double3(0.0, 0.0, 0.0);
	//*mm = make_double3(0.0, 0.0, 0.0);
	//double cdist = 0.0;
	//double im = mass[id];
	//double4 ipos = make_double4(pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	//double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	//double3 iomega = make_double3(omega[id].x, omega[id].y, omega[id].z);
	//double3 unit = make_double3(0.0, 0.0, 0.0);
	//double3 cpt = make_double3(0.0, 0.0, 0.0);
	//double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	//cdist = particle_cylinder_contact_detection(cy, ipos, unit, cpt, id);
	//double3 si = cpt - mp;
	//double3 cy2cp = cpt - cy->origin;
	//double3 Ft = make_double3(0.0, 0.0, 0.0);
	//double3 Fn = make_double3(0.0, 0.0, 0.0);
	//double3 M = make_double3(0.0, 0.0, 0.0);
	///*if (cdist > 0)
	//{
	//	double rcon = ipos.w - 0.5 * cdist;
	//	double3 dv = cy->vel + cross(cy->omega, cy2cp) - (ivel + cross(iomega, ipos.w * unit));
	//	device_force_constant c = getConstant(
	//		TCM, ipos.w, 0, im, 0, cp->Ei, cp->Ej,
	//		cp->pri, cp->prj, cp->Gi, cp->Gj,
	//		cp->rest, cp->fric, cp->rfric, cp->sratio);
	//	switch (TCM)
	//	{
	//	case 0: HMCModel(
	//		c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
	//		dv, unit, Ft, Fn, M);
	//		break;
	//	case 1:
	//		DHSModel(
	//			c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
	//			dv, unit, Ft, Fn, M);
	//		break;
	//	}
	//}*/
	//double3 sum_f = Fn + Ft;
	//force[id] += make_double3(sum_f.x, sum_f.y, sum_f.z);
	//moment[id] += make_double3(M.x, M.y, M.z);
	//mf[id] = -(Fn);
	//mm[id] = cross(si, -Fn);
}

template <typename T, unsigned int blockSize>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n)
{
	/*extern*/ __shared__ T sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	T mySum;// = make_double3(0, 0, 0);;
	mySum.x = 0.0;
	mySum.y = 0.0;
	mySum.z = 0.0;
	//sdata[tid] = make_double3(0, 0, 0);

	while (i < n)
	{
		//sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
		mySum += g_idata[i];
		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];
		i += gridSize;
	}
	sdata[tid] = mySum;
	__syncthreads();
	if ((blockSize >= 512) && (tid < 256)) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
	if ((blockSize >= 256) && (tid < 128)) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
	if ((blockSize >= 128) && (tid <  64)) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
	if ((blockSize >= 64) && (tid < 32)){ sdata[tid] = mySum = mySum + sdata[tid + 32]; } __syncthreads();
	if ((blockSize >= 32) && (tid < 16)){ sdata[tid] = mySum = mySum + sdata[tid + 16]; } __syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();

	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__device__ bool checkConcave(device_triangle_info* dpi, unsigned int* tid, unsigned int kid, double4* dsph, unsigned int cnt)
{
	double3 p1 = make_double3(0, 0, 0);
	double3 u1 = make_double3(0, 0, 0);
	double3 p2 = make_double3(dsph[kid].x, dsph[kid].y, dsph[kid].z);
	double3 u2 = dpi[kid].N;
	for (unsigned int i = 0; i < cnt; i++){
		unsigned int id = tid[i];
		p1 = make_double3(dsph[id].x, dsph[id].y, dsph[id].z);
		u1 = dpi[id].N;
		double3 p2p1 = p2 - p1;
		double chk1 = dot(p2p1, u1);
		double chk2 = dot(-p2p1, u2);
		if (chk1 > 0 && chk2 > 0)
		{
			tid[cnt++] = id;
			return true;
		}
	}
	return false;
}

template<int TCM>
__global__ void particle_polygonObject_collision_kernel(
	device_triangle_info* dpi, double4* dsph, device_mesh_mass_info* dpmi,
	double4 *pos, double3 *vel, double3 *omega, double3 *force, double3 *moment,
	double* mass, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property *cp, unsigned int np)
{
	//unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	////unsigned int np = _np;
	//if (id >= dcte.np)
	//	return;

	//double cdist = 0.0;
	//double im = mass[id];
	//double3 ipos = make_double3(pos[id].x, pos[id].y, pos[id].z);
	//double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	//double3 iomega = make_double3(omega[id].x, omega[id].y, omega[id].z);
	//double3 unit = make_double3(0.0, 0.0, 0.0);
	//int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	////double3 cpt = make_double3(0.0, 0.0, 0.0);
	////double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	//double ir = pos[id].w;
	//double3 M = make_double3(0, 0, 0);
	//int3 neighbour_pos = make_int3(0, 0, 0);
	//uint grid_hash = 0;
	//double3 Fn = make_double3(0, 0, 0);
	//double3 Ft = make_double3(0, 0, 0);
	//double3 sum_force = make_double3(0, 0, 0);
	//double3 sum_moment = make_double3(0, 0, 0);
	//unsigned int start_index = 0;
	//unsigned int end_index = 0;
	//for (int z = -1; z <= 1; z++){
	//	for (int y = -1; y <= 1; y++){
	//		for (int x = -1; x <= 1; x++){
	//			neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
	//			grid_hash = calcGridHash(neighbour_pos);
	//			start_index = cstart[grid_hash];
	//			if (start_index != 0xffffffff){
	//				end_index = cend[grid_hash];
	//				for (unsigned int j = start_index; j < end_index; j++){
	//					unsigned int k = sorted_index[j];
	//					if (k >= dcte.np)
	//					{
	//						k -= dcte.np;
	//						//printf("%d", k);
	//						double3 distVec;
	//						double dist;
	//						unsigned int pidx = dpi[k].id;
	//						device_contact_property cmp = cp[pidx];
	//						device_mesh_mass_info pmi = dpmi[pidx];
	//						double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir);
	//						double3 po2cp = cpt - pmi.origin;
	//						//double3 si = cpt - pmi.origin;
	//						distVec = ipos - cpt;
	//						dist = length(distVec);
	//						cdist = ir - dist;
	//						Fn = make_double3(0.0, 0.0, 0.0);
	//						if (cdist > 0)
	//						{
	//							double3 qp = dpi[k].Q - dpi[k].P;
	//							double3 rp = dpi[k].R - dpi[k].P;
	//							double rcon = ir - 0.5 * cdist;
	//							unit = -cross(qp, rp);// -dpi[k].N;
	//							unit = unit / length(unit);
	//							double3 dv = dcte.rollingCondition ? pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, ir * unit)) : pmi.vel - ivel;
	//							device_force_constant c = getConstant(
	//								TCM, ir, 0, im, 0, cmp.Ei, cmp.Ej,
	//								cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
	//								cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
	//						/*	switch (TCM)
	//							{
	//							case 0:
	//								HMCModel(
	//									c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
	//									dv, unit, Ft, Fn, M);
	//								break;
	//							case 1:
	//								DHSModel(
	//									c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
	//									dv, unit, Ft, Fn, M);
	//								break;
	//							}*/
	//							//double3 sum_f = Fn;
	//							sum_force += Fn + Ft;
	//							sum_moment += M;
	//							//force[id] += Fn + Ft;
	//							//moment[id] += make_double3(M.x, M.y, M.z);
	//							dpmi[pidx].force += -(Fn + Ft);// +make_double3(1.0, 5.0, 9.0);
	//							dpmi[pidx].moment += -cross(po2cp, Fn + Ft);
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	//force[id] += sum_force;
	//moment[id] += sum_moment;
}

__device__ double3 toGlobal(double3& v, double4& ep)
{
	double3 r0 = make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
	double3 r1 = make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
	double3 r2 = make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
	return make_double3
		(
		r0.x * v.x + r0.y * v.y + r0.z * v.z,
		r1.x * v.x + r1.y * v.y + r1.z * v.z,
		r2.x * v.x + r2.y * v.y + r2.z * v.z
		);
}

__global__ void updatePolygonObjectData_kernel(
	device_mesh_mass_info *dpmi, double* vList,
	double4* sphere, device_triangle_info* dpi, unsigned int ntriangle)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = ntriangle;
	if (id >= np)
		return;
	int s = id * 9;
	int mid = dpi[id].id;
	double3 pos = dpmi[mid].origin;
	double4 ep = dpmi[mid].ep;
	double4 sph = sphere[id];
	double3 P = make_double3(vList[s + 0], vList[s + 1], vList[s + 2]);
	double3 Q = make_double3(vList[s + 3], vList[s + 4], vList[s + 5]);
	double3 R = make_double3(vList[s + 6], vList[s + 7], vList[s + 8]);
	P = pos + toGlobal(P, ep);
	Q = pos + toGlobal(Q, ep);
	R = pos + toGlobal(R, ep);
	double3 V = Q - P;
	double3 W = R - P;
	double3 N = cross(V, W);
	N = N / length(N);
	double3 M1 = 0.5 * (Q + P);
	double3 M2 = 0.5 * (R + P);
	double3 D1 = cross(N, V);
	double3 D2 = cross(N, W);
	double t;
	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	}
	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	{
		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	}
	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	{
		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	}
	double3 ctri = M1 + t * D1;
	sphere[id] = make_double4(ctri.x, ctri.y, ctri.z, sph.w);
	dpi[id].P = P;
	dpi[id].Q = Q;
	dpi[id].R = R;
}