#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include <stdio.h>
#include <stdlib.h>

__constant__ device_dem_parameters cte;


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

__device__ double3 toLocal(double3& v, double4& ep)
{
	double3 r0 = make_double3(2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5), 2.0 * (ep.y*ep.z - ep.x*ep.w), 2.0 * (ep.y*ep.w + ep.x*ep.z));
	double3 r1 = make_double3(2.0 * (ep.y*ep.z + ep.x*ep.w), 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5), 2.0 * (ep.z*ep.w - ep.x*ep.y));
	double3 r2 = make_double3(2.0 * (ep.y*ep.w - ep.x*ep.z), 2.0 * (ep.z*ep.w + ep.x*ep.y), 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5));
	return make_double3
	(
		r0.x * v.x + r1.x * v.y + r2.x * v.z,
		r0.y * v.x + r1.y * v.y + r2.y * v.z,
		r0.z * v.x + r1.z * v.y + r2.z * v.z
	);
}

__device__ double3 toAngularVelocity(double4& e, double4& d)
{
	double3 o = 2.0 * make_double3(
		-e.y * d.x + e.x * d.y - e.w * d.z + e.z * d.w,
		-e.z * d.x + e.w * d.y + e.x * d.z - e.y * d.w,
		-e.w * d.x - e.z * d.y + e.y * d.z + e.x * d.w
	);
	return o;
}

__device__ double3 toEulerGlobalMoment(double4& e, double4& d)
{
	double3 o = 0.5 * make_double3(
		-e.y * d.x + e.x * d.y - e.w * d.z + e.z * d.w,
		-e.z * d.x + e.w * d.y + e.x * d.z - e.y * d.w,
		-e.w * d.x - e.z * d.y + e.y * d.z + e.x * d.w
	);
	return o;
}

__device__
uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (cte.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (cte.grid_size.y - 1);
	gridPos.z = gridPos.z & (cte.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, cte.grid_size.y), cte.grid_size.x) + __umul24(gridPos.y, cte.grid_size.x) + gridPos.x;
}

// calculate position in uniform grid
__device__
int3 calcGridPos(double3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - cte.world_origin.x) / cte.cell_size);
	gridPos.y = floor((p.y - cte.world_origin.y) / cte.cell_size);
	gridPos.z = floor((p.z - cte.world_origin.z) / cte.cell_size);
	return gridPos;
}

__device__
double4 calculate_uceom(double3& J, double4& ep, double4& ev, double3& n_prime)
{
	double4 lhs[4] = { 0, };
	lhs[0].x = 2.0 * J.x * (-ep.y); lhs[0].y = 2.0 * J.x * ep.x;	lhs[0].z = 2.0 * J.x * ep.w;	lhs[0].w = 2.0 * J.x * (-ep.z);
	lhs[1].x = 2.0 * J.y * (-ep.z); lhs[1].y = 2.0 * J.y * (-ep.w); lhs[1].z = 2.0 * J.y * ep.x;	lhs[1].w = 2.0 * J.y * ep.y;
	lhs[2].x = 2.0 * J.z * (-ep.w); lhs[2].y = 2.0 * J.z * ep.z;	lhs[2].z = 2.0 * J.z * (-ep.y); lhs[2].w = 2.0 * J.z * ep.x;
	lhs[3].x = ep.x;				lhs[3].y = ep.y;				lhs[3].z = ep.z;				lhs[3].w = ep.w;
	double3 T = make_double3(
		J.x * (-ep.y * ev.x + ep.x * ev.y + ep.w * ev.z - ep.z * ev.w),
		J.y * (-ep.z * ev.x - ep.w * ev.y + ep.x * ev.z + ep.y * ev.w),
		J.z * (-ep.w * ev.x + ep.z * ev.y - ep.y * ev.z + ep.x * ev.w));
	double4 LH0 = make_double4(
		4.0 * (-ev.y * T.x - ev.z * T.y - ev.w * T.z),
		4.0 * (ev.x * T.x - ev.w * T.y + ev.z * T.z),
		4.0 * (ev.w * T.x + ev.x * T.y - ev.y * T.z),
		4.0 * (-ev.z * T.x + ev.y * T.y + ev.x * T.z));
	double3 LH = make_double3(
		(-ep.y * LH0.x + ep.x * LH0.y + ep.w * LH0.z - ep.z * LH0.w),
		(-ep.z * LH0.x - ep.w * LH0.y + ep.x * LH0.z + ep.y * LH0.w),
		(-ep.w * LH0.x + ep.z * LH0.y - ep.y * LH0.z + ep.x * LH0.w));
	double4 r0 = make_double4(LH.x, LH.y, LH.z, dot(ev, ev));
	double4 rhs = make_double4(n_prime.x, n_prime.y, n_prime.z, 0.0) - r0;

	double det =
		lhs[0].x*lhs[1].y*lhs[2].z*lhs[3].w + lhs[0].x*lhs[1].z*lhs[2].w*lhs[3].y + lhs[0].x*lhs[1].w*lhs[2].y*lhs[3].z -
		lhs[0].x*lhs[1].w*lhs[2].z*lhs[3].y - lhs[0].x*lhs[1].z*lhs[2].y*lhs[3].w - lhs[0].x*lhs[1].y*lhs[2].w*lhs[3].z -
		lhs[0].y*lhs[1].x*lhs[2].z*lhs[3].w - lhs[0].z*lhs[1].x*lhs[2].w*lhs[3].y - lhs[0].w*lhs[1].x*lhs[2].y*lhs[3].z +
		lhs[0].w*lhs[1].x*lhs[2].z*lhs[3].y + lhs[0].z*lhs[1].x*lhs[2].y*lhs[3].w + lhs[0].y*lhs[1].x*lhs[2].w*lhs[3].z +
		lhs[0].y*lhs[1].z*lhs[2].x*lhs[3].w + lhs[0].z*lhs[1].w*lhs[2].x*lhs[3].y + lhs[0].w*lhs[1].y*lhs[2].x*lhs[3].z -
		lhs[0].w*lhs[1].z*lhs[2].x*lhs[3].y - lhs[0].z*lhs[1].y*lhs[2].x*lhs[3].w - lhs[0].y*lhs[1].w*lhs[2].x*lhs[3].z -
		lhs[0].y*lhs[1].z*lhs[2].w*lhs[3].x - lhs[0].z*lhs[1].w*lhs[2].y*lhs[3].x - lhs[0].w*lhs[1].y*lhs[2].z*lhs[3].x +
		lhs[0].w*lhs[1].z*lhs[2].y*lhs[3].x + lhs[0].z*lhs[1].y*lhs[2].w*lhs[3].x + lhs[0].y*lhs[1].w*lhs[2].z*lhs[3].x;
	matrix44d o;
	o.a00 = lhs[1].y*lhs[2].z*lhs[3].w + lhs[1].z*lhs[2].w*lhs[3].y + lhs[1].w*lhs[2].y*lhs[3].z - lhs[1].w*lhs[2].z*lhs[3].y - lhs[1].z*lhs[2].y*lhs[3].w - lhs[1].y*lhs[2].w*lhs[3].z;
	o.a01 = -lhs[0].y*lhs[2].z*lhs[3].w - lhs[0].z*lhs[2].w*lhs[3].y - lhs[0].w*lhs[2].y*lhs[3].z + lhs[0].w*lhs[2].z*lhs[3].y + lhs[0].z*lhs[2].y*lhs[3].w + lhs[0].y*lhs[2].w*lhs[3].z;
	o.a02 = lhs[0].y*lhs[1].z*lhs[3].w + lhs[0].z*lhs[1].w*lhs[3].y + lhs[0].w*lhs[1].y*lhs[3].z - lhs[0].w*lhs[1].z*lhs[3].y - lhs[0].z*lhs[1].y*lhs[3].w - lhs[0].y*lhs[1].w*lhs[3].z;
	o.a03 = -lhs[0].y*lhs[1].z*lhs[2].w - lhs[0].z*lhs[1].w*lhs[2].y - lhs[0].w*lhs[1].y*lhs[2].z + lhs[0].w*lhs[1].z*lhs[2].y + lhs[0].z*lhs[1].y*lhs[2].w + lhs[0].y*lhs[1].w*lhs[2].z;

	o.a10 = -lhs[1].x*lhs[2].z*lhs[3].w - lhs[1].z*lhs[2].w*lhs[3].x - lhs[1].w*lhs[2].x*lhs[3].z + lhs[1].w*lhs[2].z*lhs[3].x + lhs[1].z*lhs[2].x*lhs[3].w + lhs[1].x*lhs[2].w*lhs[3].z;
	o.a11 = lhs[0].x*lhs[2].z*lhs[3].w + lhs[0].z*lhs[2].w*lhs[3].x + lhs[0].w*lhs[2].x*lhs[3].z - lhs[0].w*lhs[2].z*lhs[3].x - lhs[0].z*lhs[2].x*lhs[3].w - lhs[0].x*lhs[2].w*lhs[3].z;
	o.a12 = -lhs[0].x*lhs[1].z*lhs[3].w - lhs[0].z*lhs[1].w*lhs[3].x - lhs[0].w*lhs[1].x*lhs[3].z + lhs[0].w*lhs[1].z*lhs[3].x + lhs[0].z*lhs[1].x*lhs[3].w + lhs[0].x*lhs[1].w*lhs[3].z;
	o.a13 = lhs[0].x*lhs[1].z*lhs[2].w + lhs[0].z*lhs[1].w*lhs[2].x + lhs[0].w*lhs[1].x*lhs[2].z - lhs[0].w*lhs[1].z*lhs[2].x - lhs[0].z*lhs[1].x*lhs[2].w - lhs[0].x*lhs[1].w*lhs[2].z;

	o.a20 = lhs[1].x*lhs[2].y*lhs[3].w + lhs[1].y*lhs[2].w*lhs[3].x + lhs[1].w*lhs[2].x*lhs[3].y - lhs[1].w*lhs[2].y*lhs[3].x - lhs[1].y*lhs[2].x*lhs[3].w - lhs[1].x*lhs[2].w*lhs[3].y;
	o.a21 = -lhs[0].x*lhs[2].y*lhs[3].w - lhs[0].y*lhs[2].w*lhs[3].x - lhs[0].w*lhs[2].x*lhs[3].y + lhs[0].w*lhs[2].y*lhs[3].x + lhs[0].y*lhs[2].x*lhs[3].w + lhs[0].x*lhs[2].w*lhs[3].y;
	o.a22 = lhs[0].x*lhs[1].y*lhs[3].w + lhs[0].y*lhs[1].w*lhs[3].x + lhs[0].w*lhs[1].x*lhs[3].y - lhs[0].w*lhs[1].y*lhs[3].x - lhs[0].y*lhs[1].x*lhs[3].w - lhs[0].x*lhs[1].w*lhs[3].y;
	o.a23 = -lhs[0].x*lhs[1].y*lhs[2].w - lhs[0].y*lhs[1].w*lhs[2].x - lhs[0].w*lhs[1].x*lhs[2].y + lhs[0].w*lhs[1].y*lhs[2].x + lhs[0].y*lhs[1].x*lhs[2].w + lhs[0].x*lhs[1].w*lhs[2].y;

	o.a30 = -lhs[1].x*lhs[2].y*lhs[3].z - lhs[1].y*lhs[2].z*lhs[3].x - lhs[1].z*lhs[2].x*lhs[3].y + lhs[1].z*lhs[2].y*lhs[3].x + lhs[1].y*lhs[2].x*lhs[3].z + lhs[1].x*lhs[2].z*lhs[3].y;
	o.a31 = lhs[0].x*lhs[2].y*lhs[3].z + lhs[0].y*lhs[2].z*lhs[3].x + lhs[0].z*lhs[2].x*lhs[3].y - lhs[0].z*lhs[2].y*lhs[3].x - lhs[0].y*lhs[2].x*lhs[3].z - lhs[0].x*lhs[2].z*lhs[3].y;
	o.a32 = -lhs[0].x*lhs[1].y*lhs[3].z - lhs[0].y*lhs[1].z*lhs[3].x - lhs[0].z*lhs[1].x*lhs[3].y + lhs[0].z*lhs[1].y*lhs[3].x + lhs[0].y*lhs[1].x*lhs[3].z + lhs[0].x*lhs[1].z*lhs[3].y;
	o.a33 = lhs[0].x*lhs[1].y*lhs[2].z + lhs[0].y*lhs[1].z*lhs[2].x + lhs[0].z*lhs[1].x*lhs[2].y - lhs[0].z*lhs[1].y*lhs[2].x - lhs[0].y*lhs[1].x*lhs[2].z - lhs[0].x*lhs[1].z*lhs[2].y;
	double m = (1.0 / det);// return (1.0 / det) * o;
	return m * make_double4(
		o.a00 * rhs.x + o.a01 * rhs.y + o.a02 * rhs.z + o.a03 * rhs.w,
		o.a10 * rhs.x + o.a11 * rhs.y + o.a12 * rhs.z + o.a13 * rhs.w,
		o.a20 * rhs.x + o.a21 * rhs.y + o.a22 * rhs.z + o.a23 * rhs.w,
		o.a30 * rhs.x + o.a31 * rhs.y + o.a32 * rhs.z + o.a33 * rhs.w);
}

__global__ void vv_update_position_kernel(
	double4* pos, double4* ep, double3* vel, double4* ev, double3* acc, double4* ea, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double4 new_ep = ep[id] + cte.dt * ev[id] + cte.half2dt * ea[id];
	ep[id] = normalize(new_ep);
	double3 _p = cte.dt * vel[id] + cte.half2dt * acc[id];
	pos[id].x += _p.x;
	pos[id].y += _p.y;
	pos[id].z += _p.z;
}

__global__ void vv_update_position_cluster_kernel(
	double4* pos, double4* cpos, double4* ep, double3* rloc,
	double3 *vel, double3* acc, double4* ev, double4* ea, 
	xClusterInformation* xci, unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	//unsigned int cid = id / neach;
	double4 cp = cpos[id];
//	double4 cep = ep[id];
	double4 w = ev[id];
	double4 wd = ea[id];
	double3 old_p = make_double3(cp.x, cp.y, cp.z);
	double3 new_p = old_p + cte.dt * vel[id] + cte.half2dt * acc[id];
	cpos[id] = make_double4(new_p.x, new_p.y, new_p.z, cp.w);
	/*double4 ev = make_double4(
		-cep.y * w.x - cep.z * w.y - cep.w * w.z,
		cep.x * w.x + cep.w * w.y - cep.z * w.z,
		-cep.w * w.x + cep.x * w.y + cep.y * w.z,
		cep.z * w.x - cep.y * w.y + cep.x * w.z
	);
	double4 Lpwd = make_double4(
		-cep.y * wd.x - cep.z * wd.y - cep.w * wd.z,
		cep.x * wd.x + cep.w * wd.y - cep.z * wd.z,
		-cep.w * wd.x + cep.x * wd.y + cep.y * wd.z,
		cep.z * wd.x - cep.y * wd.y + cep.x * wd.z
	);
	double4 ea = 0.5 * Lpwd - 0.25 * dot(w, w) * cep;*/
	double4 new_ep = ep[id] + cte.dt * w + cte.half2dt * wd;
	new_ep = normalize(new_ep);
	ep[id] = new_ep;
	//printf("po : %.16f, %.16f, %.16f\n", new_p.x, new_p.y, new_p.z);
	//printf("ep : %.16f, %.16f, %.16f, %.16f\n", new_ep.x, new_ep.y, new_ep.z, new_ep.w);
	unsigned int sid = sbegin + id * neach;
	for (unsigned int j = 0; j < neach; j++)
	{
		//unsigned int cid = id * 3;
		double3 m_pos = new_p + toGlobal(rloc[seach + j], new_ep);
		pos[sid + j] = make_double4(m_pos.x, m_pos.y, m_pos.z, pos[sid + j].w);
	}
}

__global__ void vv_update_velocity_kernel(
	double3* vel,
	double3* acc,
	double4* ep,
	double4* ev,
	double4* ea,
	double3* force,
	double3* moment,
	double* mass,
	double3* iner,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double m = mass[id];
	double3 v = vel[id];
	//double3 L = acc[id];
	double4 e = ep[id];
	double4 av = ev[id];
	//double3 aa = alpha[id];
	double3 a = (1.0 / m) * (force[id] + m * cte.gravity);
	double3 J = iner[id];// make_double3(0, 0, 0);
	double3 n_prime = toLocal(moment[id], e);
	double4 m_ea = calculate_uceom(J, e, av, n_prime);
	/*if (id == 0)
	{
		printf("[%d] force : [%e, %e, %e], moment : [%e, %e, %e], m_ea : [%e, %e, %e, %e]\n", id, force[id].x, force[id].y, force[id].z, moment[id].x, moment[id].y, moment[id].z, m_ea.x, m_ea.y, m_ea.z, m_ea.w);
	}*/
	
	v += 0.5 * cte.dt * (acc[id] + a);
	av = av + 0.5 * cte.dt * (ea[id] + m_ea);
	force[id] = make_double3(0.0, 0.0, 0.0); 
	moment[id] = make_double3(0.0, 0.0, 0.0);
	vel[id] = v;
	ev[id] = av;
	acc[id] = a;
	ea[id] = m_ea;
	//printf("[%d] velocity : [%e, %e, %e], ang. velocity : [%e, %e, %e]\n", id, v.x, v.y, v.z, av.x, av.y, av.z);
}

__global__ void vv_update_cluster_velocity_kernel(
	double4* cpos,
	double4* ep,
	double3* vel,
	double3* acc,
	double4* ev,
	double4* ea,
	double3* force,
	double3* moment,
	double3* rloc,
	double* mass,
	double3* iner,
	xClusterInformation* xci,
	unsigned int np)
{
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	//double4 cp = cpos[id];
	double m = mass[id];
	double3 v = vel[id];
	double3 a = acc[id];
	double4 av = ev[id];
	double4 aa = ea[id];
	double4 e = ep[id];
	double inv_m = 1.0 / m;
	//double3 in = (1.0 / iner[id]) * moment[id];
	double3 in = iner[id];
	double3 F = make_double3(0, 0, 0);
	double3 T = make_double3(0, 0, 0);
	unsigned int sid = sbegin + id * neach;
	for (unsigned int j = 0; j < neach; j++)
	{
		//double3 _F = ;
		F += force[sid + j];
		T += moment[sid + j];
		force[sid + j] = make_double3(0, 0, 0);
		moment[sid + j] = make_double3(0, 0, 0);
	}
	F += m * cte.gravity;
	double3 n_prime = toLocal(T, e);
	double4 m_ea = calculate_uceom(in, e, av, n_prime);

	v += 0.5 * cte.dt * a;
	av = av + 0.5 * cte.dt * aa;
	a = inv_m * F;
	aa = m_ea;
	vel[id] = v + 0.5 * cte.dt * a;
	ev[id] = av + 0.5 * cte.dt * aa;
	acc[id] = a;
	ea[id] = aa;
}


__global__ void calculateHashAndIndex_kernel(
	unsigned int* hash, unsigned int* index, double4* pos, unsigned int sid, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;

	volatile double4 p = pos[id];

	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	//if(_hash >= cte.ncell)
	//printf("%d hash number : %d\n",sid + id, _hash);
	hash[sid + id] = _hash;
	index[sid + id] = sid + id;
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

	//unsigned int tnp = ;// cte.np + cte.nsphere;

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
	double ir, double jr, double im, double jm,
	double iE, double jE, double ip, double jp,
	double iG, double jG, double rest,
	double fric, double s_fric, double rfric, double sratio)
{
	device_force_constant dfc = { 0, 0, 0, 0, 0, 0, 0,0,0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp * jp) + jE * (1 - ip * ip));
	double Seq = (2.0 * (2.0 - ip) * (1.0 + ip) / iE) + (2.0 * (2.0 - jp) * (1.0 + jp) / jE);
	Seq = 1.0 / Seq;
	double lne = log(rest);
	double beta = -lne * sqrt(1.0 / (lne*lne + M_PI * M_PI));
	dfc.eq_m = Meq;
	dfc.eq_r = Req;
	dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	dfc.vn = beta;
	dfc.ks = cte.contact_model == 1 ? 8.0 * Seq : dfc.kn * sratio;
	dfc.vs = cte.contact_model == 1 ? 2.0 * sqrt(5.0 / 6.0) * beta : dfc.vn * sratio;
	dfc.mu = fric;
	dfc.mu_s = s_fric;
	dfc.ms = rfric;
	//printf("eq_m = %e\n", dfc.eq_m);
	//printf("eq_r = %e\n", dfc.eq_r);
	//printf("kn = %e\n", dfc.kn);
	//printf("vn = %e\n", dfc.vn);
	//printf("ks = %e\n", dfc.ks);
	//printf("vs = %e\n", dfc.vs);
	//printf("mu = %e\n", dfc.mu);
	//printf("mu_s = %e\n", dfc.mu_s);
	//printf("ms = %e\n", dfc.ms);
	//switch (tcm)
	//{
	//case 0: {
	//	double Geq = (iG * jG) / (iG*(2 - jp) + jG * (2 - ip));
	//	double ln_e = log(rest);
	//	double xi = ln_e / sqrt(ln_e * ln_e + M_PI * M_PI);
	//	dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	//	dfc.vn = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.kn * Meq);
	//	dfc.ks = 8.0 * Geq * sqrt(Req);
	//	dfc.vs = -2.0 * sqrt(5.0 / 6.0) * xi * sqrt(dfc.ks * Meq);
	//	dfc.mu = fric;
	//	dfc.mu_s = s_fric;
	//	dfc.ms = rfric;
	//	break;
	//}
	//case 1: {
	//	//printf("rest : %f, Meq : %f", rest, Meq);
	//	double beta = (M_PI / log(rest));
	//	dfc.kn = (4.0 / 3.0) * Eeq * sqrt(Req);
	//	dfc.vn = sqrt((4.0 * Meq * dfc.kn) / (1.0 + beta * beta));
	//	dfc.ks = dfc.kn * sratio;
	//	dfc.vs = dfc.vn * sratio;
	//	dfc.mu = fric;
	//	dfc.mu_s = s_fric;
	//	dfc.ms = rfric;
	//	break;
	//}
	//}

	// 	dfc.kn = /*(16.f / 15.f)*sqrt(er) * eym * pow((T)((15.f * em * 1.0f) / (16.f * sqrt(er) * eym)), (T)0.2f);*/ (4.0f / 3.0f)*sqrt(er)*eym;
	// 	dfc.vn = sqrt((4.0f*em * dfc.kn) / (1 + beta * beta));
	// 	dfc.ks = dfc.kn * ratio;
	// 	dfc.vs = dfc.vn * ratio;
	// 	dfc.mu = fric;
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
	double cf = 0.0;
	if (coh) {
		double Req = rj ? (ri * rj) / (ri + rj) : ri/*(ri * ri) / (ri + ri)*/;
		double Eeq = 1.0 / (((1.0 - pri * pri) / Ei) + ((1.0 - prj * prj) / Ej));// (Ei * Ej) / (Ei*(1.0 - prj * prj) + Ej * (1.0 - pri * pri));
		double c0 = 3.0 * coh * M_PI * Req;
		double eq = 2.0 * c0 * Fn + c0 * c0;
		if (eq <= 0)
		{
			Fn = -0.5 * c0;
		}			
		
		double a3 = (3.0 * Req) * (Fn + c0 + sqrt(abs(2.0 * c0 * Fn + c0 * c0))) / (4.0 * Eeq);
		cf = -sqrt(8.0 * M_PI * coh * Eeq * a3);
	}
	return cf;
}

__device__ double limit_cohesion_depth(
	double ir, double jr, double iE, double jE, double ip, double jp, double coh)
{
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = 1.0 / (((1.0 - ip * ip) / iE) + ((1.0 - jp * jp) / jE));
	double c1 = (M_PI * M_PI * coh * coh * Req) / (Eeq * Eeq);
	double gs = -(3.0 / 4.0) * pow(c1, 1.0 / 3.0);
	return gs;
}

__device__ double JKR_seperation_force(double ir, double jr, double coh)
{
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double cf = -(3.0 / 2.0) * M_PI * coh * Req;
	return cf;
}

__device__ void HMCModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double cdist, double3 iomega, double& _ds, double& dots,
	double3 dv, double3 unit, double3& Ft, double3& Fn)
{
	// 	if (coh && cdist < 1.0E-8)
	// 		return;

	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = -cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn + fdn);
	// 	if ((fsn + fca + fdn) < 0 && ir)
	// 		return;
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e) {
		double3 s_hat = (e / mag_e);
		double s_dot = dot(dv, s_hat);
		double ds = _ds + cte.dt * (s_dot + dots);
		_ds = ds;
		dots = s_dot;
		double S_t = c.ks * sqrt(c.eq_r * cdist);
		Ft = min(S_t * ds + c.vs * sqrt(S_t * c.eq_m), c.mu * length(Fn)) * s_hat;
	}
}

__device__ void DHSModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double cdist, double3 iomega, double& _ds, double& dots,
	double3 dv, double3 unit, double3& Ft, double3& Fn/*, double3& M*/)
{
	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = -cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn + fdn);
	Fn = (fsn + fca + fdn) * unit;
	double3 e = dv - dot(dv, unit) * unit;
	double mag_e = length(e);
	if (mag_e)
	{
		double3 sh = e / mag_e;
		double s_dot = dot(dv, sh);
		double ds = _ds + cte.dt * (s_dot + dots);
		_ds = ds;
		dots = s_dot;
		Ft = min(c.ks * ds + c.vs * (dot(dv, sh)), c.mu * length(Fn)) * sh;
	}
}

__device__ void calculate_previous_rolling_resistance(
	double rf, double ir, double jr, double3 rc, double3 Fn, double3 Ft, double& Mr, double3& Tmax)
{
	Tmax += /*Tmax +=*/ cross(rc, Fn + Ft);

	double Rij = jr ? ir * jr / (ir + jr) : ir;
	Mr += Rij * rf * length(Fn);
}

__global__ void calcluate_clusters_contact_kernel(
	double4* pos, double4* cpos, double4* ep, double3* vel,
	double4* ev, double3* force,
	double3* moment, double* mass, double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, double2* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, xClusterInformation* xci, unsigned int np
)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2P_COUNT];
	double2 p_tsd[MAX_P2P_COUNT];
	unsigned int sid = id * MAX_P2P_COUNT;
	for (unsigned int i = 0; i < MAX_P2P_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	unsigned int old_count = pair_count[id];
	unsigned int cid = (id / neach);
	double4 icpos = cpos[cid];
	double4 ipos = pos[id];
	double4 jpos = make_double4(0, 0, 0, 0);
	double3 ivel = vel[cid];
	double3 jvel = make_double3(0, 0, 0);
	double3 iomega = toAngularVelocity(ep[cid], ev[cid]);
	double3 jomega = make_double3(0, 0, 0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));

	double ir = ipos.w; double jr = 0;
	double im = mass[cid]; double jm = 0;
	double3 Ft = make_double3(0, 0, 0);
	double3 Fn = make_double3(0, 0, 0);// [id] * cte.gravity;
	double3 M = make_double3(0, 0, 0);
	double3 sumF = make_double3(0, 0, 0);
	double3 sumM = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int new_count = sid;
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff) {
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++) {
						unsigned int k = sorted_index[j];
						unsigned int di = id >= k ? id - k : k - id;
						if (id == k || k >= np || (di <= neach))
							continue;
						unsigned int ck = (k / neach);
						jpos = pos[k]; jvel = vel[ck]; jomega = toAngularVelocity(ep[ck], ev[ck]);
						jr = jpos.w; jm = mass[ck];
						double4 jcpos = cpos[ck];
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						if (cdist > 0) {
							double2 sd = make_double2(0.0, 0.0);
							for (unsigned int i = 0; i < old_count; i++)
							{
								//unsigned int oid = sid + i;
								if (p_pair_id[i] == k)
								{
									sd = p_tsd[i];
									break;
								}
							}
							//double rcon = ir - 0.5 * cdist;							
							double3 unit = rp / dist;
							double3 cpt = make_double3(ipos.x, ipos.y, ipos.z) + ir * unit;
							double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
							double3 dcpr_j = cpt - make_double3(jcpos.x, jcpos.y, jcpos.z);
							//double3 rc = ir * unit;
							double3 rv = jvel + cross(jomega, dcpr_j) - (ivel + cross(iomega, dcpr));
							device_force_constant c = getConstant(
								ir, jr, im, jm, cp->Ei, cp->Ej,
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
							switch (cte.contact_model)
							{
							case 1:	HMCModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, rv, unit, Ft, Fn); break;
							case 0:	DHSModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, rv, unit, Ft, Fn); break;
							}
							calculate_previous_rolling_resistance(
								cp->rfric, ir, jr, dcpr, Fn, Ft, res, tma);
							sumF += Fn + Ft;
							sumM += cross(dcpr, Fn + Ft);
							tsd[new_count] = sd;
							pair_id[new_count] = k;
							new_count++;
						}
					}
				}
			}
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
	/*if (new_count - sid > MAX_P2P_COUNT)
		printf("The total of contact with other particle is over(%d)\n.", new_count - sid);*/

	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

__global__ void calculate_p2p_kernel(
	double4* pos, double4* ep, double3* vel,
	double4* ev, double3* force,
	double3* moment, double* mass, double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, double2* tsd,
	unsigned int* sorted_index, unsigned int* cstart,
	unsigned int* cend, device_contact_property* cp, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2P_COUNT];
	double2 p_tsd[MAX_P2P_COUNT];
	unsigned int sid = id * MAX_P2P_COUNT;
	for (unsigned int i = 0; i < MAX_P2P_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int old_count = pair_count[id];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double4 jpos = make_double4(0, 0, 0, 0);
	double3 ivel = vel[id];
	double3 jvel = make_double3(0, 0, 0);
	double3 iomega = toAngularVelocity(ep[id], ev[id]);
	double3 jomega = make_double3(0, 0, 0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));

	double ir = ipos.w; double jr = 0;
	double im = mass[id]; double jm = 0;
	double3 Ft = make_double3(0, 0, 0);
	double3 Fn = make_double3(0, 0, 0);// [id] * cte.gravity;
	double3 M = make_double3(0, 0, 0);
	double3 sumF = make_double3(0, 0, 0);
	double3 sumM = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	unsigned int new_count = sid;
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff) {
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++) {
						unsigned int k = sorted_index[j];
						if (id == k || k >= np)
							continue;
						jpos = pos[k]; jvel = vel[k]; jomega = toAngularVelocity(ep[k], ev[k]);
						jr = jpos.w; jm = mass[k];
						double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
						double dist = length(rp);
						double cdist = (ir + jr) - dist;
						double coh_s = 0;
						if (cp->coh)
							coh_s = limit_cohesion_depth(ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
					//	printf("cdist : %f, coh_s : %f\n", cdist, coh_s);
						double2 sd = make_double2(0.0, 0.0);
						double3 unit = rp / dist;
						double3 cpt = ipos3 + ir * unit;
						//printf("contact_dist : %f\n", cdist);
						if (cdist > 0) {
							for (unsigned int i = 0; i < old_count; i++)
								if (p_pair_id[i] == k){ sd = p_tsd[i]; break; }
							//double rcon = ir - 0.5 * cdist;
							
							double3 dcpr = cpt - ipos3;// make_double3(ipos.x, ipos.y, ipos.z);
							double3 dcpr_j = cpt - make_double3(jpos.x, jpos.y, jpos.z);
							double3 rv = jvel + cross(jomega, dcpr_j) - (ivel + cross(iomega, dcpr));
							//printf("relative velocity : [%f, %f, %f]\n", rv.x, rv.y, rv.z);
							device_force_constant c = getConstant(ir, jr, im, jm, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
						
							switch (cte.contact_model)
							{
							case 1: HMCModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, rv, unit, Ft, Fn); break;
							case 0: DHSModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, rv, unit, Ft, Fn); break;
							}
		
							calculate_previous_rolling_resistance(cp->rfric, ir, jr, dcpr, Fn, Ft, res, tma);
							sumF += Fn + Ft;

							
							sumM += cross(dcpr, Fn + Ft);
							tsd[new_count] = sd;
							pair_id[new_count] = k;
							new_count++;
						}
						else if (cdist <= 0 && abs(cdist) < abs(coh_s))
						{
						//	printf("cdist : %f, coh_s : %f\n", cdist, coh_s);
							double f = JKR_seperation_force(ir, jr, cp->coh);
							double cf = cohesionForce(ir,jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
							//printf("f : %f, cf : %f\n", f, cf);
							sumF = sumF - cf * unit;
							tsd[new_count] = sd;
							pair_id[new_count] = k;
							new_count++;
						}
					}
				}
			}
		}
	}
	//printf("[%d]particle forces : [%e, %e, %e]\n", id, sumF.x, sumF.y, sumF.z);
	//printf("[%d]particle moment : [%e, %e, %e]\n", id, sumM.x, sumM.y, sumM.z);
	force[id] += sumF;

	moment[id] += sumM;
	//printf("new_count, sid - %d, %d", new_count - sid);
	if (new_count - sid > MAX_P2P_COUNT)
		printf("The total of contact with other particle is over(%d)\n.", new_count - sid);
	
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

__device__ double particle_plane_contact_detection(
	device_plane_info* pe, double3& xp, double3& wp, double3& u, double r)
{
	double a_l1 = pow(wp.x - pe->l1, 2.0);
	double b_l2 = pow(wp.y - pe->l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r * r;

	//double h = 0;
	//if (abs(wp.z) < 1.5 * r && (wp.x > 0 && wp.x < pe->l1) && (wp.y > 0 && wp.y < pe->l2)) {
		
///	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr) {
		double3 Xsw = xp - pe->xw;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr) {
		double3 Xsw = xp - pe->w2;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe->l1 && wp.y > pe->l2 && (a_l1 + b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe->w3;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe->l2 && (sqa + b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe->w4;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe->l1) && wp.y < 0 && (sqb + sqc) < sqr) {
		double3 Xsw = xp - pe->xw;
		double3 wj_wi = pe->w2 - pe->xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->l1) && wp.y > pe->l2 && (b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe->w4;
		double3 wj_wi = pe->w3 - pe->w4;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x < 0 && (sqr + sqc) < sqr) {
		double3 Xsw = xp - pe->xw;
		double3 wj_wi = pe->w4 - pe->xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe->l2) && wp.x > pe->l1 && (a_l1 + sqc) < sqr) {
		double3 Xsw = xp - pe->w2;
		double3 wj_wi = pe->w3 - pe->w2;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe->l1) && (wp.y > 0 && wp.y < pe->l2))
	{
		double3 dp = xp - pe->xw;
		double3 uu = pe->uw / length(pe->uw);
		int pp = -sign(dot(dp, pe->uw));// dp.dot(pe->UW()));
		u = -uu;
		double collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}
	return 0;
}

__global__ void cluster_plane_contact_kernel(
	device_plane_info *plane, device_body_info* dbi,
	double* fx, double* fy, double* fz, double* mx, double* my, double* mz,
	double4* pos, double4* cpos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment,
	device_contact_property *cp, double* mass,
	double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, 
	double2* tsd, xClusterInformation* xci, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[3];
	double2 p_tsd[3];
	unsigned int sid = id * 3;
	for (unsigned int i = 0; i < 3; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	unsigned int cid = id / neach;
	unsigned int old_count = pair_count[id];
	double m = mass[cid];
	fx[id] = fy[id] = fz[id] = 0.0;// dbf[id] = make_double3(0, 0, 0);
	mx[id] = my[id] = mz[id] = 0.0;// dbm[id] = make_double3(0, 0, 0);
	double4 icpos = cpos[cid];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double r = ipos.w;
	double3 ivel = vel[cid];
	double3 iomega = toAngularVelocity(ep[cid], ev[cid]);

	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);

	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	unsigned int k = plane->id;
	unsigned int new_count = sid + (k ? old_count : 0);
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);

	double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - plane->xw;
	double3 unit = make_double3(0, 0, 0);
	double3 wp = make_double3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));
	double3 m_force = make_double3(0, 0, 0);
	double3 cpt = ipos3 + r * unit;
	double3 dcpr_j = cpt - dbi->pos;
	double coh_s = 0;
	double cdist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
	double2 sd = make_double2(0.0, 0.0);
	if (cp->coh)
		coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
	if (cdist > 0) {

		for (unsigned int i = 0; i < 3; i++)
			if (p_pair_id[i] == k) { sd = p_tsd[i];	break; }
		//double rcon = r - 0.5 * cdist;
		double3 cpt = ipos3 + r * unit;
		double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
		//double3 rc = r * unit;
		double3 oj = toAngularVelocity(dbi->ep, dbi->ed);
		double3 dv = dbi->vel + cross(oj, dcpr_j) - (ivel + cross(iomega, dcpr));
		device_force_constant c = getConstant(
			r, 0.0, m, dbi->mass, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->s_fric, cp->fric, cp->rfric, cp->sratio);
		switch (cte.contact_model)
		{
		case 0:	DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
		case 1: HMCModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
		}
		calculate_previous_rolling_resistance(
			cp->rfric, r, 0, dcpr, Fn, Ft, res, tma);
		m_force = Fn + Ft;
		sumF += m_force;
		sumM += cross(dcpr, m_force);
		//printf("po : %.16f, %.16f, %.16f\n", ipos3.x, ipos3.y, ipos3.z);
		//printf("dv : %.16f, %.16f, %.16f\n", dv.x, dv.y, dv.z);
		//printf("w : %.16f, %.16f, %.16f\n", iomega.x, iomega.y, iomega.z);
		//printf("dc : %.16f, %.16f, %.16f\n", dcpr.x, dcpr.y, dcpr.z);
		//printf("ev : %.16f, %.16f, %.16f, %.16f\n", ev[cid].x, ev[cid].y, ev[cid].z, ev[cid].w);
		/*printf("Fn : %.16f, %.16f, %.16f\n", Fn.x, Fn.y, Fn.z);
		printf("Ft : %.16f, %.16f, %.16f\n", Ft.x, Ft.y, Ft.z);*/
		//dbf[id] += -m_force;
		fx[id] += -m_force.x;
		fy[id] += -m_force.y;
		fz[id] += -m_force.z;
		double3 m_moment = -cross(dcpr_j, m_force);
		//dbm[id] += -cross(dcpr_j, m_force);
		mx[id] += m_moment.x;
		my[id] += m_moment.y;
		mz[id] += m_moment.z;
		tsd[new_count] = sd;
		pair_id[new_count] = k;
		new_count++;
	}
	else if (cdist <= 0 && abs(cdist) < abs(coh_s))
	{
		double f = JKR_seperation_force(r, 0, cp->coh);
		double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
		//printf("f : %f, cf : %f\n", f, cf);
		m_force = -cf * unit;
		sumF += m_force;
		fx[id] += -m_force.x;
		fy[id] += -m_force.y;
		fz[id] += -m_force.z;
		double3 m_moment = -cross(dcpr_j, m_force);
		//dbm[id] += -cross(dcpr_j, m_force);
		mx[id] += m_moment.x;
		my[id] += m_moment.y;
		mz[id] += m_moment.z;
		tsd[new_count] = sd;
		pair_id[new_count] = k;
		new_count++;
	}
//	}

	//printf("dbf[%d] : [%e, %e, %e]\n", id, dbf[id].x, dbf[id].y, dbf[id].z);
	force[id] += sumF;
	moment[id] += sumM;
	pair_count[id] = new_count - id * 3;
	tmax[id] += tma;
	rres[id] += res;
}

__global__ void plane_contact_force_kernel(
	device_plane_info *plane, device_body_info* dbi, device_contact_property *cp,
	double* fx, double* fy, double* fz, double* mx, double* my, double* mz,
	double4* pos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment, double* mass,
	double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, double2* tsd, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2PL_COUNT];
	double2 p_tsd[MAX_P2PL_COUNT];
	//device_body_info db[cte.nplane] = { 0, };
	unsigned int sid = id * MAX_P2PL_COUNT;
	for (unsigned int i = 0; i < MAX_P2PL_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	fx[id] = fy[id] = fz[id] = 0.0;// make_double3(0, 0, 0);
	mx[id] = my[id] = mz[id] = 0.0;// dbm[id] = make_double3(0, 0, 0);
	//for(unsigned int i = 0; i < cte.nplane)
	unsigned int old_count = pair_count[id];
	double m = mass[id];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double r = ipos.w;
	unsigned int k = plane->id;
	double3 ivel = vel[id];
	double3 iomega = toAngularVelocity(ep[id], ev[id]);

	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);

	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	unsigned int new_count = sid + (k ? old_count : 0);
	
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	//for (unsigned int k = 0; k < cte.nplane; k++)
	//{
		double coh_s = 0;
		//device_plane_info pl = plane[k];
		//device_body_info bi = dbi[pl.mid];
		//
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - plane->xw;
		double3 unit = make_double3(0, 0, 0);
		double3 wp = make_double3(dot(dp, plane->u1), dot(dp, plane->u2), dot(dp, plane->uw));
		double cdist = particle_plane_contact_detection(plane, ipos3, wp, unit, r);
		if (cp->coh)
			coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
		
		double2 sd = make_double2(0.0, 0.0);
		double3 m_force = make_double3(0, 0, 0);
		double3 cpt = ipos3 + r * unit;
		double3 dcpr_j = cpt - dbi->pos;
		if (cdist > 0) {
			printf("plane body pos\n");
			
			for (unsigned int i = 0; i < 3; i++)
				if (p_pair_id[i] == k){ sd = p_tsd[i]; break; }
			double3 dcpr = cpt - ipos3;	
			double3 oj = toAngularVelocity(dbi->ep, dbi->ed);
			double3 dv = dbi->vel + cross(oj, dcpr_j) - (ivel + cross(iomega, dcpr));
			//printf("body_vel : [%f, %f, %f]\n", bi.vel.x, bi.vel.y, bi.vel.z);
			device_force_constant c = getConstant(r, 0.0, m, 0.0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
			switch (cte.contact_model)
			{
			case 0: DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
			case 1: HMCModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
			}
			
			
			calculate_previous_rolling_resistance(cp->rfric, r, 0, dcpr, Fn, Ft, res, tma);
			//printf("kn : %f, cn : %f, ks : %f, cs : %f", c.kn, c.vn, c.ks, c.vs);
			m_force = Fn + Ft;
	
			sumF += m_force;
			sumM += cross(dcpr, m_force);
			fx[id] += -m_force.x;// dbf[id] += -m_force;
			fy[id] += -m_force.y;
			fz[id] += -m_force.z;
	
			double3 bmoment = -cross(dcpr_j, m_force);
			mx[id] += bmoment.x;// -cross(dcpr_j, m_force);
			my[id] += bmoment.y;
			mz[id] += bmoment.z;
			tsd[new_count] = sd;
			pair_id[new_count] = k;
			new_count++;
		}
		else if(cdist <= 0 && abs(cdist) < abs(coh_s))
		{
			//printf("plane seperation cohesion contact. - %f", cdist);
			double f = JKR_seperation_force(r, 0, cp->coh);
			double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
			//printf("f : %f, cf : %f\n", f, cf);
			m_force = -cf * unit;
			sumF += m_force;
			fx[id] += -m_force.x;// dbf[id] += -m_force;
			fy[id] += -m_force.y;
			fz[id] += -m_force.z;
			//printf("[%d] plane force : [%e, %e, %e]\n", id, m_force.x, m_force.y, m_force.z);
			//printf("[%d][%d] summa force : [%e, %e, %e]\n", id, dbf[id].x, dbf[id].y, dbf[id].z);
			double3 bmoment = -cross(dcpr_j, m_force);
			mx[id] += bmoment.x;// -cross(dcpr_j, m_force);
			my[id] += bmoment.y;
			mz[id] += bmoment.z;
			tsd[new_count] = sd;
			pair_id[new_count] = k;
			new_count++;
		}
		
	//}
	//printf("sumF = [%f, %f, %f]\n", sumF.x, sumF.y, sumF.z);
	//printf("sumM = [%f, %f, %f]\n", sumM.x, sumM.y, sumM.z);
	force[id] += sumF;
	moment[id] += sumM;

	if (new_count - id * 3 > 3)
		printf("The total of contact with plane is over(%d)\n.", new_count - sid);
	pair_count[id] = new_count - id * 3;
	//f("idx %d : %d\n", id, new_count - sid);
	tmax[id] += tma;
	rres[id] += res;

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

__device__ void particle_cylinder_contact_force(
	device_cylinder_info* dci,
	device_contact_property* cp,
	device_body_info* dpmi,
	double* fx,
	double* fy,
	double* fz,
	double* mx,
	double* my,
	double* mz,
	double3& ipos,
	double3& ivel,
	double3& iomega,
	double3& cpt,
	double3& unit,
	double cdist,
	unsigned int id,
	unsigned int pid,
	unsigned int old_count,
	unsigned int* p_pair_id,
	double2* p_tsd,
	unsigned int* pair_id,
	double2* tsd,
	double r,
	double m,
	double3& sum_force,
	double3& sum_moment,
	double& res,
	double3& tma,
	double coh_s,
	unsigned int& new_count)
{
	double2 sd = make_double2(0.0, 0.0);
	//device_body_info dbi = dpmi[dci->mid];
	double3 po2cp = cpt - dpmi->pos;

	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	double3 omega = toAngularVelocity(dpmi->ep, dpmi->ed);
	for (unsigned int i = 0; i < MAX_P2CY_COUNT; i++)
		if (p_pair_id[i] == pid) { sd = p_tsd[i]; break; }
	double3 rc = cpt - ipos;// r * unit;
	double3 dv = dpmi->vel + cross(omega, po2cp) - (ivel + cross(iomega, rc));

	device_force_constant c = getConstant(r, 0, m, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
	//printf("%f, %f, %f, %f\n", c.kn, c.ks, c.vn, c.vs);
	switch (cte.contact_model)
	{
	case 0:	DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	case 1: HMCModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	}

	calculate_previous_rolling_resistance(cp->rfric, r, 0, rc, Fn, Ft, res, tma);
	//printf("Fn = [%e, %e, %e]\n", Fn.x, Fn.z, Fn.y);
	//printf("Ft = [%e, %e, %e]\n", Ft.x, Ft.z, Ft.y);
	double3 m_force = Fn + Ft;
	sum_force += m_force;
	sum_moment += cross(rc, Fn + Ft);
	fx[id] += -m_force.x;// dbf[id] += -m_force;
	fy[id] += -m_force.y;
	fz[id] += -m_force.z;
	//printf("[%d] plane force : [%e, %e, %e]\n", id, m_force.x, m_force.y, m_force.z);
	//printf("[%d][%d] summa force : [%e, %e, %e]\n", id, dbf[id].x, dbf[id].y, dbf[id].z);
	double3 bmoment = -cross(po2cp, Fn + Ft);
	mx[id] += bmoment.x;// -cross(dcpr_j, m_force);
	my[id] += bmoment.y;
	mz[id] += bmoment.z;

	//dbf[id] += -(Fn + Ft);// +make_double3(1.0, 5.0, 9.0);
	//dbm[id] += -cross(po2cp, Fn + Ft);
	//printf("sumF000 = [%e, %e, %e]\n", dbf[id].x, dbf[id].z, dbf[id].y);
	//printf("sumM000 = [%e, %e, %e]\n", dbm[id].x, dbm[id].z, dbm[id].y);
	tsd[new_count] = sd;
	pair_id[new_count] = pid;
	new_count++;
	//printf("Sum Force : [%f, %f, %f]\n", Fn.x, Fn.y, Fn.z);
}

__device__ double particle_cylinder_contact_detection(
	device_cylinder_info* cy, 
	device_body_info* bi,
	double3& p, 
	double3& u,  
	double r,
	bool& isInnerContact)
{
	double dist = -1.0;
	double3 cp = make_double3(0.0, 0.0, 0.0);
	double3 cyl_pos = bi->pos;// c_ptr->Position();
	//printf("cylinder_position : [%f, %f, %f]\n", cyl_pos.x, cyl_pos.y, cyl_pos.z);
	double4 cyl_ep = bi->ep;// c_ptr->EulerParameters();
	//printf("cylinder_ep : [%f, %f, %f, %f]\n", cyl_ep.x, cyl_ep.y, cyl_ep.z, cyl_ep.w);
	double3 cyl_base = cyl_pos + toGlobal(cy->pbase, cyl_ep);
	double3 cyl_top = cyl_pos + toGlobal(cy->ptop, cyl_ep);// cinfo.ptop);
	double3 ab = cyl_top - cyl_base;// make_double3(cy->ptop.x - cy->pbase.x, cy->ptop.y - cy->pbase.y, cy->ptop.z - cy->pbase.z);
	//double3 p = make_double3(pt.x, pt.y, pt.z);
	double t = dot(p - cyl_base, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	if (t >= 0 && t <= 1) {
		_cp = cyl_base + t * ab;
		dist = length(p - _cp);
		if (dist == 0)
		{
			isInnerContact = true;
			return 0;
		}
		double gab = 0;
		u = (_cp - p) / dist;
	//	cp = _cp - cy.len_rr.z * u;
		if (dist < cy->len_rr.z)
		{
			isInnerContact = true;
			u = -u;
			gab = dist + r - cy->len_rr.y;
		}
		else
			gab = cy->len_rr.y + r - dist;
		return gab;
	}
	else {
		_cp = cyl_base + t * ab;
		dist = length(p - _cp);
		double thick = cy->thickness;
		int one = thick ? 1 : 0;
		if (dist < cy->len_rr.z + thick && dist > one * cy->len_rr.z) {
			double3 OtoCp = bi->pos - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
		//	cp = _cp - cy.len_rr.z * u;
			return cy->len_rr.x * 0.5 + r - OtoCp_;
		}
	/*	double3 A_1 = makeTFM_1(cy->ep);
		double3 A_2 = makeTFM_2(cy->ep);
		double3 A_3 = makeTFM_3(cy->ep);*/
		double3 _at = p - cyl_top;
		double3 at = toLocal(_at, cyl_ep);
		//double r = length(at);
		cp = cyl_top;
		if (abs(at.y) > cy->len_rr.x) {
			_at = p - cyl_base;
			at = toLocal(_at, cyl_ep);
			cp = cyl_base;
		}
		double pi = atan(at.x / at.z);
		if (pi < 0 && at.z < 0) {
			_cp.x = cy->len_rr.z * sin(-pi);
		}
		else if (pi > 0 && at.x < 0 && at.z < 0) {
			_cp.x = cy->len_rr.z * sin(-pi);
		}
		else {
			_cp.x = cy->len_rr.z * sin(pi);
		}
		_cp.z = cy->len_rr.z * cos(pi);
		if (at.z < 0 && _cp.z > 0) {
			_cp.z = -_cp.z;
		}
		else if (at.z > 0 && _cp.z < 0) {
			_cp.z = -_cp.z;
		}
		_cp.y = 0.;
		cp = cp + toGlobal(_cp, cyl_ep);

		double3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < r) {
			return r - dist;
		}
	}
	return -1.0;
}

__device__ double particle_cylinder_inner_base_or_top_contact_detection(
	device_cylinder_info* cy, 
	device_body_info* bi,
	double3& p, double3 & u, double3 & cp, unsigned int empty, double r)
{
	//isInnerContact = false;
	double dist = -1.0;
	double3 cyl_pos = bi->pos;// c_ptr;// ->Position();
	double4 cyl_ep = bi->ep;// .c_ptr->EulerParameters();
	double3 cyl_base = cyl_pos + toGlobal(cy->pbase, cyl_ep);
	double3 cyl_top = cyl_pos + toGlobal(cy->ptop, cyl_ep);
	double3 ab = cyl_top - cyl_base;
	//double3 p = new_vector3d(pt.x, pt.y, pt.z);
	double t = dot(p - cyl_base, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	double gab = -1.0;
	_cp = cyl_base + t * ab;
	dist = length(p - _cp);
	if (empty == 1 && t > 0.6)
		return 0.0;	
	if (empty == 2 && t < 0.4)
		return 0.0;
	//printf("empty : %d\n", empty);
	if (dist < cy->len_rr.z) {
		double3 OtoCp = cyl_pos - _cp;
		double OtoCp_ = length(OtoCp);
		u = -OtoCp / OtoCp_;
		cp = p + r * u;// _cp - hci.len_rr.z * u;
		gab = r + OtoCp_ - cy->len_rr.x * 0.5;// +r - OtoCp_;
	}

	return gab;
}

__global__ void cluster_cylinder_contact_force_kernel(
	device_cylinder_info *cy, device_body_info* bi,
	double* fx, double* fy, double* fz, double* mx, double* my, double* mz,
	double4* pos, double4* cpos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment, xClusterInformation* xci, 
	device_contact_property *cp,
	double* mass, double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id,
	double2* tsd, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2CY_COUNT];
	double2 p_tsd[MAX_P2CY_COUNT];
	unsigned int sid = id * MAX_P2CY_COUNT;
	for (unsigned int i = 0; i < MAX_P2CY_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	unsigned int cid = id / neach;
	unsigned int old_count = pair_count[id];
	fx[id] = fy[id] = fz[id] = 0.0;// make_double3(0, 0, 0);
	mx[id] = my[id] = mz[id] = 0.0;// dbm[id] = make_double3(0, 0, 0);
	//unsigned int old_count = pair_count[id];
	double m = mass[cid];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double4 icpos = cpos[cid];
	double r = ipos.w;
	double3 ivel = vel[cid];
	double3 iomega = toAngularVelocity(ep[cid], ev[cid]);

	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);

	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	unsigned int k = cy->id;
	unsigned int new_count = sid + (k ? old_count : 0);

	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	device_body_info dbi = { 0, };
	double cdist = 0;
	double3 cpt = make_double3(0.0, 0.0, 0.0);

	bool isInnerContact = false;
	double coh_s = 0;
	double3 unit = make_double3(0, 0, 0);
	cdist = particle_cylinder_contact_detection(cy, bi, ipos3, unit, r, isInnerContact);
	//double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, r);
	if (cp->coh)
		coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
	double2 sd = make_double2(0.0, 0.0);
	cpt = ipos3 + r * unit;
	double3 icpos3 = make_double3(icpos.x, icpos.y, icpos.z);
	if (cdist > 0)
	{
		particle_cylinder_contact_force(
			cy, cp, bi, fx, fy, fz, mx, my, mz, 
			icpos3, ivel, iomega, cpt, unit, cdist, id, k,
			old_count, p_pair_id, p_tsd, pair_id, tsd, r, m,
			sumF, sumM, res, tma, coh_s, new_count);
	}
	else if (cdist <= 0 && abs(cdist) < abs(coh_s))
	{
		//printf("plane seperation cohesion contact. - %f", cdist);
		double f = JKR_seperation_force(r, 0, cp->coh);
		double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
		//printf("f : %f, cf : %f\n", f, cf);
		double3 m_force = -cf * unit;
		sumF = sumF + m_force;
		fx[id] += -m_force.x;
		fy[id] += -m_force.y;
		fz[id] += -m_force.z;
		double3 m_moment = -cross(cpt - dbi.pos, m_force);
		mx[id] += m_moment.x;// -cross(cpt - dbi.pos, m_force);
		my[id] += m_moment.y;
		mz[id] += m_moment.z;
		tsd[new_count] = sd;
		pair_id[new_count] = k;
		new_count++;
	}
	if (isInnerContact)
	{
		cdist = particle_cylinder_inner_base_or_top_contact_detection(
			cy, bi, ipos3, unit, cpt, cy->empty_part, r);
		if (cdist > 0)
		{

			particle_cylinder_contact_force(
				cy, cp, bi, fx, fy, fz, mx, my, mz, 
				icpos3, ivel, iomega, cpt, unit, cdist, id, k + 1000,
				old_count, p_pair_id, p_tsd, pair_id, tsd, r, m,
				sumF, sumM, res, tma, coh_s, new_count);

		}
		else if (cdist <= 0 && abs(cdist) < abs(coh_s))
		{
			double f = JKR_seperation_force(r, 0, cp->coh);
			double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
			double3 m_force = -cf * unit;
			sumF = sumF + m_force;
			fx[id] += -m_force.x;
			fy[id] += -m_force.y;
			fz[id] += -m_force.z;
			//dbm[id] += -cross(cpt - dbi.pos, m_force);
			double3 m_moment = -cross(cpt - bi->pos, m_force);
			mx[id] += m_moment.x;// -cross(cpt - dbi.pos, m_force);
			my[id] += m_moment.y;
			mz[id] += m_moment.z;
			tsd[new_count] = sd;
			pair_id[new_count] = k + 1000;
			new_count++;
		}
	}

	force[id] += sumF;
	moment[id] += sumM;
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

__global__ void cylinder_contact_force_kernel(
	device_cylinder_info *cy, device_body_info* bi,
	double *fx, double* fy, double* fz, double* mx, double* my, double* mz,
	double4* pos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment, device_contact_property *cp,
	double* mass, double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id,
	double2* tsd, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2CY_COUNT];
	double2 p_tsd[MAX_P2CY_COUNT];
	unsigned int sid = id * MAX_P2CY_COUNT;
	for (unsigned int i = 0; i < MAX_P2CY_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	fx[id] = fy[id] = fz[id] = 0.0;// make_double3(0, 0, 0);
	mx[id] = my[id] = mz[id] = 0.0;// dbm[id] = make_double3(0, 0, 0);
	unsigned int old_count = pair_count[id];
	unsigned int k = cy->id;
	double m = mass[id];
	double4 ipos = pos[id];
	double3 ipos3 = make_double3(ipos.x, ipos.y, ipos.z);
	double r = ipos.w;
	double3 ivel = vel[id];
	double3 iomega = toAngularVelocity(ep[id], ev[id]);

	double3 Fn = make_double3(0, 0, 0);
	double3 Ft = make_double3(0, 0, 0);
	double3 M = make_double3(0, 0, 0);

	double3 sumF = make_double3(0.0, 0.0, 0.0);
	double3 sumM = make_double3(0.0, 0.0, 0.0);
	unsigned int new_count = sid + (k ? old_count : 0);

	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	//device_body_info dbi = { 0, };
	double cdist = 0;
	double3 cpt = make_double3(0.0, 0.0, 0.0);

	bool isInnerContact = false;
	double coh_s = 0;
	//device_cylinder_info dci = cy[k];
	//if (dci.ismoving)
	//dbi = bi[k];
	//device_contact_property dcp = cp[k];
	double3 unit = make_double3(0, 0, 0);
	cdist = particle_cylinder_contact_detection(cy, bi, ipos3, unit, r, isInnerContact);
	//double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, r);
	if (cp->coh)
		coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
	double2 sd = make_double2(0.0, 0.0);
	cpt = ipos3 + r * unit;
	if (cdist > 0)
	{
		//printf("cylinder contact distance : %f\n", cdist);
		particle_cylinder_contact_force(
			cy, cp, bi, fx, fy, fz, mx, my, mz, 
			ipos3, ivel, iomega, cpt, unit, cdist, id, k,
			old_count, p_pair_id, p_tsd, pair_id, tsd, r, m,
			sumF, sumM, res, tma, coh_s, new_count);
	}
	else if (cdist <= 0 && abs(cdist) < abs(coh_s))
	{
		//printf("...............cohesion0");
		//printf("plane seperation cohesion contact. - %f", cdist);
		double f = JKR_seperation_force(r, 0, cp->coh);
		double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
		//printf("f : %f, cf : %f\n", f, cf);
		double3 m_force = -cf * unit;
		sumF = sumF + m_force;
		//dbf[id] += -m_force;
		fx[id] += -m_force.x;
		fy[id] += -m_force.y;
		fz[id] += -m_force.z;
		//dbm[id] += -cross(cpt - dbi.pos, m_force);
		double3 m_moment = -cross(cpt - bi->pos, m_force);
		mx[id] += m_moment.x;
		my[id] += m_moment.y;
		mz[id] += m_moment.z;
		tsd[new_count] = sd;
		pair_id[new_count] = k;
		/*printf("sumF111 = [%e, %e, %e]\n", dbf[id].x, dbf[id].z, dbf[id].y);
		printf("sumM111 = [%e, %e, %e]\n", dbm[id].x, dbm[id].z, dbm[id].y);*/
		new_count++;
	}
	if (isInnerContact)
	{
		cdist = particle_cylinder_inner_base_or_top_contact_detection(
			cy, bi, ipos3, unit, cpt, cy->empty_part, r);
		if (cdist > 0)
		{

			particle_cylinder_contact_force(
				cy, cp, bi, fx, fy, fz, mx, my, mz, 
				ipos3, ivel, iomega, cpt, unit, cdist, id, k + 1000,
				old_count, p_pair_id, p_tsd, pair_id, tsd, r, m,
				sumF, sumM, res, tma, coh_s, new_count);

		}
		else if (cdist <= 0 && abs(cdist) < abs(coh_s))
		{
			//printf("...............cohesion1");
			double f = JKR_seperation_force(r, 0, cp->coh);
			double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
			double3 m_force = -cf * unit;
			sumF = sumF + m_force;
			fx[id] += -m_force.x;
			fy[id] += -m_force.y;
			fz[id] += -m_force.z;
		/*	dbm[id] += -cross(cpt - dbi.pos, m_force);*/
			double3 m_moment = -cross(cpt - bi->pos, m_force);
			mx[id] += m_moment.x;
			my[id] += m_moment.y;
			mz[id] += m_moment.z;
			tsd[new_count] = sd;
			pair_id[new_count] = k + 1000;
			/*printf("sumF222 = [%e, %e, %e]\n", dbf[id].x, dbf[id].z, dbf[id].y);
			printf("sumM222 = [%e, %e, %e]\n", dbm[id].x, dbm[id].z, dbm[id].y);*/
			new_count++;
		}
	}
	
	force[id] += sumF;
	moment[id] += sumM;
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

template <unsigned int blockSize>
__global__ void reduce6(double3 *g_idata, double3 *g_odata, unsigned int n)
{
	extern __shared__ double3 sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	//double3 mySum;// = make_double3(0, 0, 0);;
	//mySum.x = 0.0;
	//mySum.y = 0.0;
	//mySum.z = 0.0;
	//sdata[tid] = make_double3(0, 0, 0);
	
	/*if(i < n)
		sdata[tid] = make_double3(0.0, 0.0, 0.0);*/
	while (i < n)
	{
		
		//sdata[tid] += g_idata[i] + g_idata[i + blockSize]; 
		/*mySum += g_idata[i];
		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];
		i += gridSize;*/
		
	//	printf("gridSize : %d", gridSize);
		sdata[tid] += g_idata[i];
		if(i + blockSize)
			sdata[tid] += g_idata[i + blockSize];
		i += gridSize;
	}
	//printf("reduce mySum0 : %e\n", sdata[tid]);
	//sdata[tid] = mySum;
	
	__syncthreads();
	if ((blockSize >= 512) && (tid < 256)) { sdata[tid] += /*mySum = mySum + */sdata[tid + 256]; } __syncthreads();
	if ((blockSize >= 256) && (tid < 128)) { sdata[tid] += /*mySum = mySum + */sdata[tid + 128]; } __syncthreads();
	if ((blockSize >= 128) && (tid < 64)) { sdata[tid] += /*mySum = mySum + */sdata[tid + 64]; } __syncthreads();
	if (tid < 32)
	{
		if (blockSize >= 64) { sdata[tid] += /*mySum = mySum + */sdata[tid + 32]; } __syncthreads();
		if (blockSize >= 32) { sdata[tid] += /*mySum = mySum + */sdata[tid + 16]; } __syncthreads();
		if (blockSize >= 16) { sdata[tid] += /*mySum = mySum + */sdata[tid + 8]; } __syncthreads();
		if (blockSize >= 8) { sdata[tid] += /*mySum = mySum + */sdata[tid + 4]; } __syncthreads();
		if (blockSize >= 4) { sdata[tid] += /*mySum = mySum + */sdata[tid + 2]; } __syncthreads();
		if (blockSize >= 2)	{ sdata[tid] += /*mySum = mySum + */sdata[tid + 1]; } __syncthreads();
	}
	
	//printf("reduce mySum1 : %e\n", mySum);
	
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];// mySum;
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
	double3 cpt = make_double3(0, 0, 0);
	//if (ct == 2)
	//{
		if (d1 <= 0.0 && d2 <= 0.0) { ct = 0; cpt = a; }
		if (d3 >= 0.0 && d4 <= d3) { ct = 0; cpt = b; }
		if (d6 >= 0.0 && d5 <= d6) { ct = 0; cpt = c; }
//	}
//	else if (ct == 1)
	//{
		if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { ct = 1; cpt = a + (d1 / (d1 - d3)) * ab; }
		if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { ct = 1; cpt = a + (d2 / (d2 - d6)) * ac; }
		if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { ct = 1; cpt = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b); }
//	}	
	//else if (ct == 0)
	//{
		if (va >= 0 && vb >= 0 && vc >= 0)
		{
			double denom = 1.0 / (va + vb + vc);
			double3 v = vb * denom * ab;
			double3 w = vc * denom * ac;
			cpt = a + v + w;
			//double _dist = pr - length(p - _cpt);
			ct = 2;
			//if (_dist > 0) return _cpt;
		}
	//}
		return cpt;
}

__device__ bool checkConcave(device_triangle_info* dti, unsigned int* tid, unsigned int kid, double4* dsph, unsigned int cnt)
{
	/*double3 p1 = make_double3(0, 0, 0);
	double3 u1 = make_double3(0, 0, 0);
	double3 p2 = make_double3(dsph[kid].x, dsph[kid].y, dsph[kid].z);
	double3 u2 = dpi[kid].N;
	for (unsigned int i = 0; i < cnt; i++) {
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
	}*/
	return false;
}

__device__ bool checkOverlab(/*int3 ctype,*/ double3 p, double3 c/*double3 u0, double3 u1*/)
{
	bool b_over0 = false;
	bool b_over1 = false;
	bool b_over2 = false;
	if (p.x >= c.x - 1e-9 && p.x <= c.x + 1e-9) b_over0 = true;
	if (p.y >= c.y - 1e-9 && p.y <= c.y + 1e-9) b_over1 = true;
	if (p.z >= c.z - 1e-9 && p.z <= c.z + 1e-9) b_over2 = true;

	//if (/*(ctype.y || ctype.z) &&*/ !b_over)
	//{
	//	if (u0.x >= u1.x - 1e-9 && u0.x <= u1.x + 1e-9)
	//		if (u0.y >= u1.y - 1e-9 && u0.y <= u1.y + 1e-9)
	//			if (u0.z >= u1.z - 1e-9 && u0.z <= u1.z + 1e-9)
	//				b_over = true;
	//}
	return (b_over0 && b_over1 && b_over2);
}

__device__ void particle_triangle_contact_force(
	device_triangle_contact_info& dtci,
	int t,
	unsigned int id,
	device_triangle_info* dti,
	device_contact_property *cp,
	device_body_info* dbi,
	double* fx,
	double* fy, 
	double* fz,
	double* mx,
	double* my,
	double* mz,
	double3& ipos,
	double3& ivel,
	double3& iomega,
	unsigned int old_count,
	unsigned int* p_pair_id,
	double2* p_tsd,
	unsigned int* pair_id,
	double2* tsd,
	double r,
	double m,
	double3& sum_force,
	double3& sum_moment,
	double& res,
	double3& tma,
	unsigned int& new_count)
{
	//device_triangle_info tri = dti[dtci.id];
	//double3 qp = tri.Q - tri.P;
	//double3 rp = tri.R - tri.P;
	double3 rp = dtci.cpt - ipos;//-cross(qp, rp);
	///unit = unit / length(unit);
	double2 sd = make_double2(0.0, 0.0);
	double3 po2cp = dtci.cpt - dbi->pos;// pmi.origin;
	double dist = length(rp);
	double3 unit = rp / dist;
	double cdist = r - dist;//r - length(ipos - dtci.cpt);
	double coh_s = 0;
	if (cp->coh)
		coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
	if (cdist <= 0 && abs(cdist) < abs(coh_s))
	{
		double f = JKR_seperation_force(r, 0, cp->coh);
		double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
		sum_force = sum_force - cf * unit;
		tsd[new_count] = sd;
		pair_id[new_count] = dtci.id;
		new_count++;
		return;
	}
	
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	double3 jomega = toAngularVelocity(dbi->ep, dbi->ed);
	for (unsigned int i = 0; i < MAX_P2MS_COUNT; i++)
		if (p_pair_id[i] == dtci.id){ sd = p_tsd[i]; break; }
	double3 rc = dtci.cpt - ipos;
	double3 dv = dbi->vel + cross(jomega, po2cp) - (ivel + cross(iomega, rc));
	device_force_constant c = getConstant(r, 0, m, dbi->mass, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
	switch (cte.contact_model)
	{
	case 0: DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	case 1: HMCModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	}
	

	calculate_previous_rolling_resistance(cp->rfric, r, 0, rc, Fn, Ft, res, tma);
	double3 m_force = Fn + Ft;
	//printf("[%d] m_force : [%e, %e, %e]\n", id, m_force.x, m_force.y, m_force.z);
	sum_force += m_force;
	sum_moment += cross(rc, m_force);
	//dbf[id] += -m_force;// +make_double3(1.0, 5.0, 9.0);
	fx[id] += -m_force.x;
	fy[id] += -m_force.y;
	fz[id] += -m_force.z;
	//dbm[id] += 
	double3 m_moment = -cross(po2cp, m_force);
	mx[id] += m_moment.x;
	my[id] += m_moment.y;
	mz[id] += m_moment.z;
	tsd[new_count] = sd;
	pair_id[new_count] = dtci.id;
	
	new_count++;
}

__device__ void cluster_triangle_contact_force(
	device_triangle_contact_info dtci,
	int t,
	unsigned int id,
	device_triangle_info* dpi,
	device_contact_property *cp,
	device_body_info* dbi,
	double* fx,
	double* fy,
	double* fz,
	double* mx,
	double* my,
	double* mz,
	double3& ipos,
	double3& icpos,
	double3& ivel,
	double3& iomega,
	unsigned int old_count,
	unsigned int* p_pair_id,
	double2* p_tsd,
	unsigned int* pair_id,
	double2* tsd,
	double r,
	double m,
	double3& sum_force,
	double3& sum_moment,
	double& res,
	double3& tma,
	unsigned int& new_count)
{
	device_triangle_info tri = dpi[dtci.id];// unsigned int pidx = dpi[k].id;
	double3 qp = tri.Q - tri.P;
	double3 rp = tri.R - tri.P;
	//	double rcon = r - 0.5 * cdist;
	double3 unit = -cross(qp, rp);
	//unsigned int pidx = dpi[dtci.id].id;
	//device_contact_property cp = cp[pidx];
	//device_body_info db = dbi[pidx];// device_mesh_mass_info pmi = dpmi[pidx];
	//double3 cpt = closestPtPointTriangle(dpi[k], ipos, r, t);
	double3 po2cp = dtci.cpt - dbi->pos;// pmi.origin;
	double cdist = r - length(ipos - dtci.cpt);
	double coh_s = 0.0;
	if (cp->coh)
		coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
	double2 sd = make_double2(0.0, 0.0);
	if (cdist <= 0 && abs(cdist) < abs(coh_s))
	{
		double f = JKR_seperation_force(r, 0, cp->coh);
		double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
		sum_force = sum_force - cf * unit;
		tsd[new_count] = sd;
		pair_id[new_count] = dtci.id;
		new_count++;
		return;
	}
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	//device_triangle_info tri = dpi[k];
	
	double3 dcpr = dtci.cpt - make_double3(icpos.x, icpos.y, icpos.z);
	//double3 dcpr_j = cpt - make_double3(jcpos.x, jcpos.y, jcpos.z);
	unit = unit / length(unit);
//	double2 sd = make_double2(0.0, 0.0);
	for (unsigned int i = 0; i < MAX_P2MS_COUNT; i++)
		if (p_pair_id[i] == dtci.id) { sd = p_tsd[i]; break; }
	//double3 rc = r * unit;
	double3 jomega = toAngularVelocity(dbi->ep, dbi->ed);
	//printf("jomega : [%e, %e, %e]\n", jomega.x, jomega.y, jomega.z);
	double3 dv = dbi->vel + cross(jomega, po2cp) - (ivel + cross(iomega, dcpr));
	device_force_constant c = getConstant(r, 0, m, dbi->mass, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->s_fric, cp->rfric, cp->sratio);
	
	switch (cte.contact_model)
	{
		case 0: DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
		case 1: HMCModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	}
	calculate_previous_rolling_resistance(cp->rfric, r, 0, dcpr, Fn, Ft, res, tma);
	double3 m_force = Fn + Ft;
	sum_force += m_force;
	sum_moment += cross(dcpr, Fn + Ft);
	
	fx[id] += -m_force.x;
	fy[id] += -m_force.y;
	fz[id] += -m_force.z;// +make_double3(1.0, 5.0, 9.0);
	double3 m_moment = -cross(po2cp, Fn + Ft);
	mx[id] += m_moment.x;// -cross(po2cp, Fn + Ft);
	my[id] += m_moment.y;
	mz[id] += m_moment.z;
	tsd[new_count] = sd;
	pair_id[new_count] = dtci.id;
	new_count++;
}

__global__ void particle_polygonObject_collision_kernel(
	device_triangle_info* dti, device_body_info* dbi, 
	double* fx, double* fy, double* fz, double* mx, double* my, double* mz,
	double4 *pos, double4 *ep, double3 *vel, double4 *ev, double3 *force, double3 *moment,
	double* mass, double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, double2* tsd, double4* dsph,
	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	device_contact_property *cp, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2MS_COUNT];
	double2 p_tsd[MAX_P2MS_COUNT];
	unsigned int sid = id * MAX_P2MS_COUNT;
	for (unsigned int i = 0; i < MAX_P2MS_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int old_count = pair_count[id];
	if (old_count > 0)
		return;
	//	double cdist = 0.0;
	double im = mass[id];
	double3 ipos = make_double3(pos[id].x, pos[id].y, pos[id].z);
	double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	double3 iomega = toAngularVelocity(ep[id], ev[id]);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = pos[id].w;
	double3 M = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;

	double3 sum_force = make_double3(0, 0, 0);
	double3 sum_moment = make_double3(0, 0, 0);
	//unsigned int new_count = sid;
	unsigned int sk = dti[0].tid;
	unsigned int new_count = sid;
	
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	int3 ctype = make_int3(0, 0, 0);
	device_triangle_contact_info ctriangle[20] = { 0, };
	device_triangle_contact_info cline[20] = { 0, };
	device_triangle_contact_info cpoint[20] = { 0, };
	unsigned int nct = 0;
	unsigned int ncl = 0;
	unsigned int ncp = 0;
	double coh_s = 0;
	
	double3 previous_line_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_point_cpt = make_double3(0.0, 0.0, 0.0);
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff) {
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++) {
						unsigned int tk = sorted_index[j];
						if (tk >= cte.np)
						{
							tk -= cte.np + sk;							
							///if (k < bindex || k >= eindex)
							//	continue;
							int t = -1;
							double3 cpt = closestPtPointTriangle(dti[tk], ipos, ir, t);
							//device_contact_property cmp = cp;
							double cdist = ir - length(ipos - cpt);
							if (cp->coh)
								coh_s = limit_cohesion_depth(ir, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
							if (cdist > 0)
							{
								/*printf("seeee : %d, %d, %d", j, start_index, end_index);
								printf("contact_type : %d\n", t);
								printf("num_p : %d\n", cte.np);
								printf("start_index : %d\n", sk);
								printf("target_sphere : %d\n", tk);
								printf("target_cdist : %e\n", cdist);
								printf("P : [%e, %e, %e]\n", dti[tk].P.x, dti[tk].P.y, dti[tk].P.z);
								printf("Q : [%e, %e, %e]\n", dti[tk].Q.x, dti[tk].Q.y, dti[tk].Q.z);
								printf("R : [%e, %e, %e]\n", dti[tk].R.x, dti[tk].R.y, dti[tk].R.z);*/
								//printf("cidst : %f, contact_type : %d\n", cdist, t);
								if (t == 1)
								{
									if (checkOverlab(previous_line_cpt, cpt))
										continue;
									previous_line_cpt = cpt;
								}
								else if (t == 0)
								{
									if (checkOverlab(previous_point_cpt, cpt))
										continue;
									previous_point_cpt = cpt;
								}
								switch (t)
								{
								case 2: ctriangle[nct++] = { tk, cpt }; break;
								case 1: cline[ncl++] = { tk, cpt }; break;
								case 0: cpoint[ncp++] = { tk, cpt }; break;
								}
							}
							else if(cdist <= 0 && abs(cdist) < abs(coh_s))
							{
								switch (t)
								{
								case 2: ctriangle[nct++] = { tk, cpt }; break;
								case 1: cline[ncl++] = { tk, cpt }; break;
								case 0: cpoint[ncp++] = { tk, cpt }; break;
								}
							}
						}
					}
				}
			}
		}
	}
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	fx[id] = fy[id] = fz[id] = 0.0;
	mx[id] = my[id] = mz[id] = 0.0;
	//printf("tlp : [%d - %d - %d]\n", nct, ncl, ncp);
	for (unsigned int k = 0; k < nct; k++)
		particle_triangle_contact_force(
			ctriangle[k], 2, id, dti, cp, dbi, 
			fx, fy, fz, mx, my, mz,
			ipos, ivel, iomega, old_count, 
			p_pair_id, p_tsd, pair_id, tsd, 
			ir, im, sum_force, sum_moment, res, tma, new_count);
	if (!nct)
	{
		//double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
		//double3 previous_unit = make_double3(0.0, 0.0, 0.0);
		for (unsigned int k = 0; k < ncl; k++)
		{
			//if (checkOverlab(previous_cpt, cline[k].cpt))
			//	continue;
			//previous_cpt = cline[k].cpt;
			particle_triangle_contact_force(
				cline[k], 1, id, dti, cp, dbi, 
				fx, fy, fz, mx, my, mz,
				ipos, ivel, iomega, old_count,
				p_pair_id, p_tsd, pair_id, tsd, ir, im,
				sum_force, sum_moment, res, tma, new_count);
		}
	}
		
	if (!nct && !ncl)
	{
		//printf("ncp : %d\n", ncp);
		//double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
		for (unsigned int k = 0; k < ncp; k++)
		{
			//if (checkOverlab(previous_cpt, cpoint[k].cpt))
			//	continue;
		//	previous_cpt = cpoint[k].cpt;
			particle_triangle_contact_force(
				cpoint[k], 0, id, dti, cp, dbi, 
				fx, fy, fz, mx, my, mz,
				ipos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
		}			
	}
		

	force[id] += sum_force;
	moment[id] += sum_moment;
	//if (length(sum_force))
		//printf("particle_force_with_mesh : [%d, %d, %d]-[%e, %e, %e]\n",nct, ncl, ncp, sum_force.x, sum_force.y, sum_force.z);
	///*if (new_count - sid > MAX_P2MS_COUNT)
	//	printf("The total of contact with triangle */is over(%d)\n.", new_count - sid);
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

__global__ void cluster_meshes_contact_kernel(
	device_triangle_info *dti, device_body_info* dbi,
	double* fx, double* fy, double* fz, double* mx, double* my, double* mz,// double3* dbf, double3* dbm,
	double4* pos, double4* cpos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment,
	device_contact_property *cp, double* mass,
	double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id,
	double2* tsd, unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
	xClusterInformation* xci, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	//unsigned int np = _np;
	if (id >= np)
		return;
	unsigned int p_pair_id[MAX_P2MS_COUNT];
	double2 p_tsd[MAX_P2MS_COUNT];
	unsigned int sid = id * MAX_P2MS_COUNT;
	for (unsigned int i = 0; i < MAX_P2MS_COUNT; i++)
	{
		p_pair_id[i] = pair_id[sid + i];
		p_tsd[i] = tsd[sid + i];
	}
	unsigned int neach = 0;
	unsigned int seach = 0;
	unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
		sbegin += xc.count * xc.neach;
	}
	unsigned int cid = (id / neach);
	unsigned int old_count = pair_count[id];
	//	double cdist = 0.0;
	double im = mass[cid];
	double4 ipos4 = pos[id];
	double4 cpos4 = cpos[cid];
	double3 ipos = make_double3(ipos4.x, ipos4.y, ipos4.z);
	double3 icpos = make_double3(cpos4.x, cpos4.y, cpos4.z);
	double3 ivel = vel[cid];
	double3 iomega = toAngularVelocity(ep[cid], ev[cid]);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
	double ir = pos[id].w;
	double3 M = make_double3(0, 0, 0);
	int3 neighbour_pos = make_int3(0, 0, 0);
	uint grid_hash = 0;

	double3 sum_force = make_double3(0, 0, 0);
	double3 sum_moment = make_double3(0, 0, 0);
	unsigned int new_count = sid;// +(bindex ? old_count : 0);

	double3 previous_line_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_point_cpt = make_double3(0.0, 0.0, 0.0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	int3 ctype = make_int3(0, 0, 0);
	device_triangle_contact_info ctriangle[20] = { 0, };
	device_triangle_contact_info cline[20] = { 0, };
	device_triangle_contact_info cpoint[20] = { 0, };
	unsigned int nct = 0;
	unsigned int ncl = 0;
	unsigned int ncp = 0;
	unsigned int sk = dti[0].id;
	double coh_s = 0;
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
				grid_hash = calcGridHash(neighbour_pos);
				start_index = cstart[grid_hash];
				if (start_index != 0xffffffff) {
					end_index = cend[grid_hash];
					for (unsigned int j = start_index; j < end_index; j++) {
						unsigned int tk = sorted_index[j];
						if (tk >= cte.np)
						{
							tk -= cte.np + sk;
							int t = -1;
							double3 cpt = closestPtPointTriangle(dti[tk], ipos, ir, t);
							device_contact_property cmp = cp[dti[tk].id];
							double cdist = ir - length(ipos - cpt);
							if (cmp.coh)
								coh_s = limit_cohesion_depth(ir, 0, cmp.Ei, cmp.Ej, cmp.pri, cmp.prj, cmp.coh);
							if (cdist > 0)
							{
								if (t == 1)
								{
									if (checkOverlab(previous_line_cpt, cpt))
										continue;
									previous_line_cpt = cpt;
								}
								else if (t == 0)
								{
									if (checkOverlab(previous_point_cpt, cpt))
										continue;
									previous_point_cpt = cpt;
								}
								switch (t)
								{
								case 2: ctriangle[nct++] = { tk, cpt }; break;
								case 1: cline[ncl++] = { tk, cpt }; break;
								case 0: cpoint[ncp++] = { tk, cpt }; break;
								}
							}
							else if (cdist <= 0 && abs(cdist) < abs(coh_s))
							{
								switch (t)
								{
								case 2: ctriangle[nct++] = { tk, cpt }; break;
								case 1: cline[ncl++] = { tk, cpt }; break;
								case 0: cpoint[ncp++] = { tk, cpt }; break;
								}
							}
						}
					}
				}
			}
		}
	}
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	//printf("tlp : [%d - %d - %d]\n", nct, ncl, ncp);
	fx[id] = fy[id] = fz[id] = 0.0;
	mx[id] = my[id] = mz[id] = 0.0;
	for (unsigned int k = 0; k < nct; k++)
		cluster_triangle_contact_force(
			ctriangle[k], 0, id, dti, cp, dbi, 
			fx, fy, fz, mx, my, mz, 
			ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if (!nct)
		for (unsigned int k = 0; k < ncl; k++)
			cluster_triangle_contact_force(
				cline[k], 1, id, dti, cp, dbi, 
				fx, fy, fz, mx, my, mz,
				ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if (!nct && !ncl)
		for (unsigned int k = 0; k < ncp; k++)
			cluster_triangle_contact_force(
				cpoint[k], 2, id, dti, cp, dbi, 
				fx, fy, fz, mx, my, mz,
				ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	for (unsigned int i = 0; i < 5; i++)
		tsd[sid + i] = make_double2(0, 0);
	force[id] += sum_force;
	moment[id] += sum_moment;
	///*if (new_count - sid > MAX_P2MS_COUNT)
	//	printf("The total of contact with triangle */is over(%d)\n.", new_count - sid);
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

//template<int TCM>
//__global__ void particle_polygonObject_collision_kernel(
//	device_triangle_info* dpi, device_mesh_mass_info* dpmi,
//	double4 *pos, double3 *vel, double3 *omega, double3 *force, double3 *moment,
//	double* mass, double3* tmax, double* rres,
//	unsigned int* pair_count, unsigned int* pair_id, double2* tsd, double4* dsph,
//	unsigned int* sorted_index, unsigned int* cstart, unsigned int* cend,
//	device_contact_property *cp, unsigned int np)
//{
//	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//	//unsigned int np = _np;
//	if (id >= np)
//		return;
//	unsigned int p_pair_id[MAX_P2MS_COUNT];
//	double2 p_tsd[MAX_P2MS_COUNT];
//	unsigned int sid = id * MAX_P2MS_COUNT;
//	for (unsigned int i = 0; i < MAX_P2MS_COUNT; i++)
//	{
//		p_pair_id[i] = pair_id[sid + i];
//		p_tsd[i] = tsd[sid + i];
//	}
//	unsigned int old_count = pair_count[id];
////	double cdist = 0.0;
//	double im = mass[id];
//	double3 ipos = make_double3(pos[id].x, pos[id].y, pos[id].z);
//	double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
//	double3 iomega = make_double3(omega[id].x, omega[id].y, omega[id].z);
//	double3 unit = make_double3(0.0, 0.0, 0.0);
//	int3 gridPos = calcGridPos(make_double3(ipos.x, ipos.y, ipos.z));
//	double ir = pos[id].w;
//	double3 M = make_double3(0, 0, 0);
//	int3 neighbour_pos = make_int3(0, 0, 0);
//	uint grid_hash = 0;
//	double3 Fn = make_double3(0, 0, 0);
//	double3 Ft = make_double3(0, 0, 0);
//	double3 sum_force = make_double3(0, 0, 0);
//	double3 sum_moment = make_double3(0, 0, 0);
//	unsigned int new_count = sid;
//	double res = 0.0;
//	double3 tma = make_double3(0.0, 0.0, 0.0);
//	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
//	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
//	unsigned int start_index = 0;
//	unsigned int end_index = 0;
//	int3 ctype = make_int3(0, 0, 0);
//	for (int z = -1; z <= 1; z++) {
//		for (int y = -1; y <= 1; y++) {
//			for (int x = -1; x <= 1; x++) {
//				neighbour_pos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
//				grid_hash = calcGridHash(neighbour_pos);
//				start_index = cstart[grid_hash];
//				if (start_index != 0xffffffff) {
//					end_index = cend[grid_hash];
//					for (unsigned int j = start_index; j < end_index; j++) {
//						unsigned int k = sorted_index[j];
//						if (k >= cte.np)
//						{
//							k -= cte.np;
////							int t = -1;
//							unsigned int pidx = dpi[k].id;
//							device_contact_property cmp = cp[pidx];
//							device_mesh_mass_info pmi = dpmi[pidx];
//							double4 jpos = dsph[k];
//							double jr = jpos.w;
//							double3 rp = make_double3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
//							double dist = length(rp);
//							double cdist = (ir + jr) - dist;
//							//double3 cpt = closestPtPointTriangle(dpi[k], ipos, ir, t);
//							
//							//double cdist = ir - length(ipos - cpt)*/;
//							Fn = make_double3(0.0, 0.0, 0.0);
//							if (cdist > 0)
//							{
//								double rcon = ir - 0.5 * cdist;
//								
//								double3 unit = rp / dist;
//								//double3 rc = ir * unit;
//								//double3 rv = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
//
//								//device_triangle_info tri = dpi[k];
//								//double3 qp = tri.Q - tri.P;
//								//double3 rp = tri.R - tri.P;
//								//double rcon = ir - 0.5 * cdist;
//								//unit = -cross(qp, rp);
//							//	unit = unit / length(unit);
//								//bool overlab = checkOverlab(ctype, previous_cpt, cpt, previous_unit, unit);
//								//printf("is overlab : %d", overlab);
//								//if (overlab)
//								//	continue;
//								double2 sd = make_double2(0.0, 0.0);
//								for (unsigned int i = 0; i < old_count; i++)
//								{
//									if (p_pair_id[i] == k)
//									{
//										sd = p_tsd[i];
//										break;
//									}
//								}
//								//*(&(ctype.x) + t) += 1;
//								//printf("index : %d - %f\n", k, dist);
//								//printf("ctype : [%d, %d, %d]\n", ctype.x, ctype.y, ctype.z);
//								//previous_cpt = cpt;
//								//previous_unit = unit;
//								//printf("ctype : [%f, %f, %f]\n", unit.x, unit.y, unit.z);
//								double3 rc = ir * unit;
//								//double3 cpt = 0.5 * (ipos + jpos);
//								double3 po2cp = (ipos + rc) - pmi.origin;
//								double3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, rc));
//								device_force_constant c = getConstant(
//									TCM, ir, 0, im, 0, cmp.Ei, cmp.Ej,
//									cmp.pri, cmp.prj, cmp.Gi, cmp.Gj,
//									cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
//								switch (TCM)
//								{
//								case 0:
//									HMCModel(
//										c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
//										dv, unit, Ft, Fn, M);
//									break;
//								case 1:
//									DHSModel(
//										c, ir, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega, sd.x, sd.y,
//										dv, unit, Ft, Fn, M);
//									break;
//								}
//								calculate_previous_rolling_resistance(
//									cmp.rfric, ir, 0, rc, Fn, Ft, res, tma);
//								sum_force += Fn + Ft;
//								sum_moment += M;
//								dpmi[pidx].force += -/*cmp.amp * */(Fn + Ft);// +make_double3(1.0, 5.0, 9.0);
//								dpmi[pidx].moment += -cross(po2cp, /*cmp.amp * */(Fn + Ft));
//								tsd[new_count] = sd;
//								pair_id[new_count] = k;
//								new_count++;
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	force[id] += sum_force;
//	moment[id] += sum_moment;
//	/*if (new_count - sid > MAX_P2MS_COUNT)
//		printf("The total of contact with triangle is over(%d)\n.", new_count - sid);*/
//	pair_count[id] = new_count - sid;
//	tmax[id] += tma;
//	rres[id] += res;
//}

__global__ void decide_rolling_friction_moment_kernel(
	double3* tmax,
	double* rres,
	double* inertia,
	double4 *ep,
	double4 *ev,
	double3 *moment,
	unsigned int np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double Tr = rres[id];
	if (!Tr)
		return;
	double3 Tmax = tmax[id];// make_double3(rfm[id].x, rfm[id].y, rfm[id].z);

	double J = inertia[id];
	double3 iomega = toAngularVelocity(ep[id], ev[id]);
	double3 _Tmax = J * cte.dt * iomega - Tmax;
	if (length(_Tmax) && Tr)
	{
		double3 _Tr = Tr * (_Tmax / length(_Tmax));
	//	printf("maximum__moment[%d] : [%e, %e, %e]\n", id, _Tmax.x, _Tmax.y, _Tmax.z);
		if (length(_Tr) >= length(_Tmax))
			_Tr = _Tmax;
		//printf("previous_moment[%d] : [%e, %e, %e]\n", id, moment[id].x, moment[id].y, moment[id].z);
		//printf("registan_moment[%d] : [%e, %e, %e]\n", id, _Tr.x, _Tr.y, _Tr.z);
		moment[id] += _Tr;
	//	printf("after____moment[%d] : [%e, %e, %e]\n", id, moment[id].x, moment[id].y, moment[id].z);
	}
	tmax[id] = make_double3(0, 0, 0);
	rres[id] = 0.0;
	//rfm[id] = make_double4(0.0, 0.0, 0.0, 0.0);
}

__global__ void decide_cluster_rolling_friction_moment_kernel(
	double3* tmax,
	double* rres,
	double3* inertia,
	double4* ep,
	double4* ev,
	double3* moment,
	xClusterInformation* xci,
	unsigned int np)
{
	unsigned int id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= np)
		return;
	double Tr = rres[id];
	if (!Tr)
		return;
	unsigned int neach = 0;
	unsigned int seach = 0;
	//unsigned int sbegin = 0;
	for (unsigned int i = 0; i < cte.nClusterObject; i++)
	{
		xClusterInformation xc = xci[i];
		if (id >= xc.sid && id < xc.sid + xc.count * xc.neach)
		{
			neach = xc.neach;
			break;
		}
		seach += xc.neach;
	}
	double3 Tmax = tmax[id];// make_double3(rfm[id].x, rfm[id].y, rfm[id].z);
	unsigned int cid = id / neach;
	double3 J = inertia[cid];
	double3 iomega = toAngularVelocity(ep[cid], ev[cid]);
	double3 J_iomega = make_double3(J.x * iomega.x, J.y * iomega.y, J.z * iomega.z);
	double3 _Tmax = cte.dt * J_iomega - Tmax;
	if (length(_Tmax) && Tr)
	{
		double3 _Tr = Tr * (_Tmax / length(_Tmax));
		if (length(_Tr) >= length(_Tmax))
			_Tr = _Tmax;
		moment[id] -= _Tr;
	}
	tmax[id] = make_double3(0, 0, 0);
	rres[id] = 0.0;
}

//__device__ double3 toGlobal(double4& ep, double3& s)
//{
//	double3 tv;
//	tv.x = 2.0 * (ep.x*ep.x + ep.y*ep.y - 0.5) * s.x + 2.0 * (ep.y*ep.z - ep.x*ep.w) * s.y + 2.0 * (ep.y*ep.w + ep.x*ep.z) * s.z;
//	tv.y = 2.0 * (ep.y*ep.z + ep.x*ep.w) * s.x + 2.0 * (ep.x*ep.x + ep.z*ep.z - 0.5) * s.y + 2.0 * (ep.z*ep.w - ep.x*ep.y) * s.z;
//	tv.z = 2.0 * (ep.y*ep.w - ep.x*ep.z) * s.x + 2.0 * (ep.z*ep.w + ep.x*ep.y) * s.y + 2.0 * (ep.x*ep.x + ep.w*ep.w - 0.5) * s.z;
//	return tv;
//}

//__device__ double3 calculate_center_of_triangle(double3& P, double3& Q, double3& R)
//{
//	double3 V = Q - P;
//	double3 W = R - P;
//	double3 N = cross(V, W);
//	//printf("V = [%f %f %f]\n", V.x, V.y, V.z);
//	//printf("W = [%f %f %f]\n", W.x, W.y, W.z);
//	N = N / length(N);
//	double3 M1 = (Q + P) / 2.0;
//	double3 M2 = (R + P) / 2.0;
//	double3 D1 = cross(N, V);
//
//	double3 D2 = cross(N, W);
//	double t;// = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
//	if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
//	{
//		t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
//	}
//	else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
//	{
//		t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
//	}
//	else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
//	{
//		t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
//	}
//	//printf("%f\n", t);
//	//printf("D1 = [%f %f %f]\n", D1.x, D1.y, D1.z);
//	return M1 + t * D1;
//}



__global__ void updateMeshObjectData_kernel(
	device_body_info *dbi, double* vList,
	double4* sphere, double3* dlocal, device_triangle_info* dpi, unsigned int ntriangle)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = ntriangle;
	if (id >= np)
		return;
	int s = id * 9;
	//int mid = dpi[id].id;
//	printf("idx(%d) : mid = %d\n", id, mid);
	double3 pos = dbi->pos;// dpmi[mid].origin;
	double4 ep = dbi->ep;// mep[mid];// dpmi[mid].ep;
	double4 sph = sphere[id];
	double3 P = make_double3(vList[s + 0], vList[s + 1], vList[s + 2]);
	double3 Q = make_double3(vList[s + 3], vList[s + 4], vList[s + 5]);
	double3 R = make_double3(vList[s + 6], vList[s + 7], vList[s + 8]);
	P = pos + toGlobal(P, ep);
	Q = pos + toGlobal(Q, ep);	
	R = pos + toGlobal(R, ep);
	//printf("idx(%d) : ep = [%f, %f, %f, %f]\n", id, ep.x, ep.y, ep.z, ep.w);
	//printf("idx(%d) : P = [%f, %f, %f]\n", id, P.x, P.y, P.z);
	//printf("idx(%d) : Q = [%f, %f, %f]\n", id, Q.x, Q.y, Q.z);
	//printf("idx(%d) : R = [%f, %f, %f]\n", id, R.x, R.y, R.z);

	//
	//double3 V = Q - P;
	//double3 W = R - P;
	//double3 N = cross(V, W);
	//N = N / length(N);
	//double3 M1 = 0.5 * (Q + P);
	//double3 M2 = 0.5 * (R + P);
	//double3 D1 = cross(N, V);
	//double3 D2 = cross(N, W);
	//double t;
	//if (abs(D1.x*D2.y - D1.y*D2.x) > 1E-13)
	//{
	//	t = (D2.x*(M1.y - M2.y)) / (D1.x*D2.y - D1.y*D2.x) - (D2.y*(M1.x - M2.x)) / (D1.x*D2.y - D1.y*D2.x);
	//}
	//else if (abs(D1.x*D2.z - D1.z*D2.x) > 1E-13)
	//{
	//	t = (D2.x*(M1.z - M2.z)) / (D1.x*D2.z - D1.z*D2.x) - (D2.z*(M1.x - M2.x)) / (D1.x*D2.z - D1.z*D2.x);
	//}
	//else if (abs(D1.y*D2.z - D1.z*D2.y) > 1E-13)
	//{
	//	t = (D2.y*(M1.z - M2.z)) / (D1.y*D2.z - D1.z*D2.y) - (D2.z*(M1.y - M2.y)) / (D1.y*D2.z - D1.z*D2.y);
	//}
	//double3 ctri = M1 + t * D1;
	double3 r_pos = pos + toGlobal(dlocal[id], ep);
	//if (id == 0)
	//{
	//	printf("%f, %f, %f\n", r_pos.x, r_pos.y, r_pos.z);
	//}
	sphere[id] = make_double4(r_pos.x, r_pos.y, r_pos.z, sph.w);
	dpi[id].P = P;
	dpi[id].Q = Q;
	dpi[id].R = R;
}

__global__ void calculate_spring_damper_force_kernel(
	double4* pos,
	double3* vel,
	double3* force,
	xSpringDamperConnectionInformation* xsdci,
	xSpringDamperConnectionData* xsdcd,
	xSpringDamperCoefficient* xsdkc)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= cte.nTsdaConnection)
		return;
	unsigned int i = xsdci[id].id;
	double4 p = pos[i];
	double3 ri = make_double3(p.x, p.y, p.z);
	double3 vi = vel[i];
	xSpringDamperConnectionInformation xsi = xsdci[id];
	for (unsigned int j = 0; j < xsi.ntsda; j++)
	{
		xSpringDamperConnectionData xsd = xsdcd[xsi.sid + j];
		xSpringDamperCoefficient kc = xsdkc[xsd.kc_id];
		double4 pj = pos[xsd.jd];
		double3 rj = make_double3(pj.x, pj.y, pj.z);
		double3 vj = vel[xsd.jd];
		double3 L = rj - ri;
		double l = length(L);
		double dl = dot(L, (vj - vi)) / l;
		double fr = kc.k * (l - xsd.init_l) + kc.c * dl;
		//printf("%d, %d, %f\n", xsd.jd, xsd.kc_id, fl[xsi.sid + j]);
		double3 Q = (fr / l) * L;
		force[i] += Q;
	}
}

__device__ double3 BMatrix_mul(double4 e, double3 s, double4 v)
{
	double4 B0 = make_double4(2 * (2 * s.x*e.x + e.z*s.z - e.w*s.y), 2 * (2 * s.x*e.y + e.w*s.z + e.z*s.y), 2 * (e.y*s.y + e.x*s.z), 2 * (e.y*s.z - e.x*s.y));
	double4 B1 = make_double4(2 * (2 * s.y*e.x - e.y*s.z + e.w*s.x), 2 * (s.x*e.z - e.x*s.z), 2 * (2 * s.y*e.z + e.w*s.z + e.y*s.x), 2 * (e.z*s.z + e.x*s.x));
	double4 B2 = make_double4(2 * (2 * s.z*e.x - e.z*s.x + e.y*s.y), 2 * (s.x*e.w + e.x*s.y), 2 * (e.w*s.y - e.x*s.x), 2 * (2 * s.z*e.w + e.z*s.y + e.y*s.x));
	return make_double3(dot(B0, v), dot(B1, v), dot(B2,v));
}

__device__ double4 BMatrix_mul_t(double4 e, double3 s, double3 v)
{
	double4 B0 = make_double4(2 * (2 * s.x*e.x + e.z*s.z - e.w*s.y), 2 * (2 * s.x*e.y + e.w*s.z + e.z*s.y), 2 * (e.y*s.y + e.x*s.z), 2 * (e.y*s.z - e.x*s.y));
	double4 B1 = make_double4(2 * (2 * s.y*e.x - e.y*s.z + e.w*s.x), 2 * (s.x*e.z - e.x*s.z), 2 * (2 * s.y*e.z + e.w*s.z + e.y*s.x), 2 * (e.z*s.z + e.x*s.x));
	double4 B2 = make_double4(2 * (2 * s.z*e.x - e.z*s.x + e.y*s.y), 2 * (s.x*e.w + e.x*s.y), 2 * (e.w*s.y - e.x*s.x), 2 * (2 * s.z*e.w + e.z*s.y + e.y*s.x));
	return make_double4(
		B0.x * v.x + B1.x * v.y + B2.x * v.z,
		B0.y * v.x + B1.y * v.y + B2.y * v.z,
		B0.z * v.x + B1.z * v.y + B2.z * v.z,
		B0.w * v.x + B1.w * v.y + B2.w * v.z);// (B1, v), dot(B2, v));
}

__global__ void calculate_spring_damper_connecting_body_force_kernel(
	double4* pos,
	double3* vel,
	double4* ep,
	double4* ev,
	double* mass,
	double3* force,
	double3* moment,
	device_tsda_connection_body_data* xsdbcd,
	xSpringDamperCoefficient* xsdkc)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= cte.nTsdaConnectionBodyData)
		return;
	device_tsda_connection_body_data cdata = xsdbcd[id];
//	device_body_info dbi = xsdbi[cdata.body_id];
	//unsigned int pid = xsdbcd[id].id;

	//unsigned int i = xsdci[id].id;
	double4 rj4 = pos[cdata.id];
	double4 ri4 = pos[cdata.body_id];
	double3 rj = make_double3(rj4.x, rj4.y, rj4.z);
	double3 ri = make_double3(ri4.x, ri4.y, ri4.z);
	double3 vj = vel[cdata.id];
	double3 vi = vel[cdata.body_id];
	double4 e = ep[cdata.body_id];
	double4 ed = ev[cdata.body_id];

	xSpringDamperCoefficient kc = xsdkc[cdata.kc_id];
	double3 L = rj - ri - toGlobal(cdata.rpos, e);
	double3 dL = vj - vi - BMatrix_mul(ed, cdata.rpos, e);
	double l = length(L);
	double dl = dot(L, dL) / l;
	double fr = kc.k * (l - cdata.init_l) + kc.c * dl;
	//printf("%d, %d, %f\n", xsd.jd, xsd.kc_id, fl[xsi.sid + j]);
	double3 Q = (fr / l) * L;
	double4 QRi = (fr / l) * BMatrix_mul_t(e, cdata.rpos, L);
	force[cdata.body_id] += Q;
	//vector3d Qi = (fr / l) * L;
	force[cdata.id] += -Q;
	//vector3d Qj = -Qi;
	moment[cdata.body_id] = moment[cdata.body_id] + toEulerGlobalMoment(e, QRi);// *L;
	/*pm->addAxialForce(Qi.x, Qi.y, Qi.z);
	pm->addEulerParameterMoment(QRi.x, QRi.y, QRi.z, QRi.w);
	f[d.ci] = Qj;*/
	//xSpringDamperConnectionInformation xsi = xsdci[id];
}