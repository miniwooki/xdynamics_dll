#include "xdynamics_parallel/xParallelDEM_decl.cuh"
#include <stdio.h>
#include <stdlib.h>

__constant__ device_dem_parameters cte;

inline __device__ int sign(double L)
{
	return L < 0 ? -1 : 1;
}

inline __device__ double dot(double3 v1, double3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __device__ double dot(double4 v1, double4 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

inline __device__ double3 operator-(double3& v1, double3& v2)
{
	return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __device__ double4 operator-(double4& v1, double4& v2)
{
	return make_double4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}

inline __device__ double4 operator+(double4& v1, double4& v2)
{
	return make_double4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

inline __device__ double3 operator-(double3& v1)
{
	return make_double3(-v1.x, -v1.y, -v1.z);
}

inline __device__ double3 operator*(double v1, double3& v2)
{
	return make_double3(v1 * v2.x, v1 * v2.y, v1 * v2.z);
}

inline __device__ double4 operator*(double v1, double4& v2)
{
	return make_double4(v1 * v2.x, v1 * v2.y, v1 * v2.z, v1 * v2.w);
}

inline __device__ double3 operator+(double3& v1, double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __host__ __device__ void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ double3 operator/(double3& v1, double v2)
{
	return make_double3(v1.x / v2, v1.y / v2, v1.z / v2);
}

inline __device__ double4 operator/(double4& v1, double v2)
{
	return make_double4(v1.x / v2, v1.y / v2, v1.z / v2, v1.w / v2);
}

inline __device__ double length(double3 v1)
{
	return sqrt(dot(v1, v1));
}

inline __device__ double length(double4 v1)
{
	return sqrt(dot(v1, v1));
}

inline __device__ double3 cross(double3 a, double3 b)
{
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __device__ double4 normalize(double4 v)
{
	return v / length(v);
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
	double* iner,
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
	double3 J = make_double3(0, 0, 0);
	if (i >= np - cte.nmp)
	{
		unsigned int sid = np - cte.nmp;
		unsigned int j = id - (np - cte.nmp);
		J.x = iner[sid + j * 3 + 0];
		J.y = iner[sid + j * 3 + 1];
		J.z = iner[sid + j * 3 + 2];
	}
	else
	{
		double in = iner[id];
		J = make_double3(in, in, in);
	}
	
	double3 n_prime = toLocal(moment[id], e);
	double4 m_ea = calculate_uceom(J, e, av, n_prime);
	/*if(length(force[id]) > 0)
		printf("[%f, %f, %f]\n", force[id].x, force[id].y, force[id].z);*/
	//double3 in = (1.0 / iner[id]) * moment[id];
	v += 0.5 * cte.dt * (acc[id] + a);
	av = av + 0.5 * cte.dt * (ea[id] + m_ea);
	force[id] = make_double3(0.0, 0.0, 0.0); 
	moment[id] = make_double3(0.0, 0.0, 0.0);
	vel[id] = v;
	ev[id] = av;
	acc[id] = a;
	ea[id] = m_ea;
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
//	printf("%f, %f, %f\n", T.x, T.y, T.z);
	//printf("%f, %f, %f, %f\n", m_ea.x, m_ea.y, m_ea.z, m_ea.w);
	/*double3 w_prime = toLocal(av, e);
	double3 Jwp = make_double3(in.x * w_prime.x, in.y * w_prime.y, in.z * w_prime.z);
	double3 tJwp = make_double3(-av.z * Jwp.y + av.y * Jwp.z, av.z * Jwp.x - av.x * Jwp.z, -av.y * Jwp.x + av.x * Jwp.y);
	double3 rhs = n_prime - tJwp;
	double3 wd_prime*/// = make_double3(rhs.x / in.x, rhs.y / in.y, rhs.z / in.z);
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
	unsigned int* hash, unsigned int* index, double4* pos, unsigned int np)
{
	unsigned id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (id >= (np)) return;

	volatile double4 p = pos[id];

	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	unsigned _hash = calcGridHash(gridPos);
	/*if(_hash >= cte.ncell)
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
	int tcm, double ir, double jr, double im, double jm,
	double iE, double jE, double ip, double jp,
	double iG, double jG, double rest,
	double fric, double rfric, double sratio)
{
	device_force_constant dfc = { 0, 0, 0, 0, 0, 0 };
	double Meq = jm ? (im * jm) / (im + jm) : im;
	double Req = jr ? (ir * jr) / (ir + jr) : ir;
	double Eeq = (iE * jE) / (iE*(1 - jp * jp) + jE * (1 - ip * ip));
	switch (tcm)
	{
	case 0: {
		double Geq = (iG * jG) / (iG*(2 - jp) + jG * (2 - ip));
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
	case 1: {
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
		double Req = rj ? (ri * rj) / (ri + rj) : ri;
		double Eeq = (Ei * Ej) / (Ei*(1 - prj * prj) + Ej * (1 - pri * pri));
		double c0 = 3.0 * coh * M_PI * Req;
		double eq = 2.0 * c0 * Fn + c0 * c0;
		if (eq <= 0)
			Fn = -0.5 * c0;
		double a3 = (3.0 * Req) * (Fn + c0 + sqrt(2.0 * c0 * Fn + c0 * c0)) / (4.0 * Eeq);
		/*double rcp = (3.0 * req * (-Fn)) / (4.0 * (1.0 / Eeq));
		double rc = pow(rcp, 1.0 / 3.0);
		double Ac = M_PI * rc * rc;
		cf = coh * Ac;*/
		cf = /*(4.0 * coh_e * a3) / (3.0 * coh_r)*/ -sqrt(8.0 * M_PI * coh * Eeq * a3);
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
// 		float fsn = -c.kn * pow(collid_dist, 1.5f);
// 		float fca = cohesionForce(ir, jr, E, E, pr, pr, coh, fsn);
// 		float fsd = c.vn * dot(relative_vel, unit);
// 		float3 single_force = (fsn + fca + fsd) * unit;
// 		//float3 single_force = (-c.kn * pow(collid_dist, 1.5f) + c.vn * dot(relative_vel, unit)) * unit;
// 		float3 single_moment = make_float3(0, 0, 0);
// 		float3 e = relative_vel - dot(relative_vel, unit) * unit;
// 		float mag_e = length(e);
// 		if (mag_e){
// 			float3 s_hat = e / mag_e;
// 			float ds = mag_e * cte.dt;
// 			float fst = -c.ks * ds;
// 			float fdt = c.vs * dot(relative_vel, s_hat);
// 			shear_force = (fst + fdt) * s_hat;
// 			if (length(shear_force) >= c.mu * length(single_force))
// 				shear_force = c.mu * fsn * s_hat;
// 			single_moment = cross(rcon * unit, shear_force);
// 			if (length(iomega)){
// 				float3 on = iomega / length(iomega);
// 				single_moment += -rfric * fsn * rcon * on;
// 			}
// 			//shear_force = min(c.ks * ds + c.vs * (dot(relative_vel, s_hat)), c.mu * length(single_force)) * s_hat;
// 			//single_moment = cross(ir * unit, shear_force);
// 		}
// 		force += single_force + shear_force;
// 		moment += single_moment;
// 	}
// 	
// 	return true;
// }

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
	if (mag_e) {
		double3 s_hat = -(e / mag_e);
		double ds = mag_e * cte.dt;
		double fst = -c.ks * ds;
		double fdt = c.vs * dot(dv, s_hat);
		Ft = (fst + fdt) * s_hat;
		if (length(Ft) >= c.mu * length(Fn))
			Ft = c.mu * fsn * s_hat;
		M = cross(ir * unit, Ft);
		if (length(iomega)) {
			double3 on = iomega / length(iomega);
			M += c.ms * fsn * rcon * on;
		}
	}
}

__device__ void DHSModel(
	device_force_constant c, double ir, double jr, double Ei, double Ej, double pri, double prj, double coh,
	double cdist, double3 iomega, double& _ds, double& dots,
	double3 dv, double3 unit, double3& Ft, double3& Fn/*, double3& M*/)
{
	double fsn = -c.kn * pow(cdist, 1.5);
	double fdn = c.vn * dot(dv, unit);
	double fca = -cohesionForce(ir, jr, Ei, Ej, pri, prj, coh, fsn);
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
		//M = cross(ir * unit, Fn + Ft);
		/*if (length(iomega)){
		double3 on = iomega / length(iomega);
		M += c.ms * fsn * rcon * on;
		}*/
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
							double rcon = ir - 0.5 * cdist;							
							double3 unit = rp / dist;
							double3 cpt = make_double3(ipos.x, ipos.y, ipos.z) + ir * unit;
							double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
							double3 dcpr_j = cpt - make_double3(jcpos.x, jcpos.y, jcpos.z);
							//double3 rc = ir * unit;
							double3 rv = jvel + cross(jomega, dcpr_j) - (ivel + cross(iomega, dcpr));
							device_force_constant c = getConstant(
								1, ir, jr, im, jm, cp->Ei, cp->Ej,
								cp->pri, cp->prj, cp->Gi, cp->Gj,
								cp->rest, cp->fric, cp->rfric, cp->sratio);
							switch (1)
							{
							case 0:
								HMCModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, rcon, cdist, iomega,
									rv, unit, Ft, Fn, M);
								break;
							case 1:
								DHSModel(
									c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj,
									cp->coh, cdist, iomega, sd.x, sd.y,
									rv, unit, Ft, Fn);
								break;
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
	tmax[id] = tma;
	rres[id] = res;
}

template <int TCM>
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
						double coh_s = 0.0;
						if (cp->coh)
							coh_s = limit_cohesion_depth(ir, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
						double2 sd = make_double2(0.0, 0.0);
						double3 unit = rp / dist;
						if (cdist > 0) {
							for (unsigned int i = 0; i < old_count; i++)
								if (p_pair_id[i] == k){ sd = p_tsd[i]; break; }
							double rcon = ir - 0.5 * cdist;
							
							double3 rc = ir * unit;
							double3 rv = jvel + cross(jomega, -jr * unit) - (ivel + cross(iomega, ir * unit));
							device_force_constant c = getConstant(TCM, ir, jr, im, jm, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->rfric, cp->sratio);
							switch (TCM)
							{
							case 0: HMCModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, rcon, cdist, iomega, rv, unit, Ft, Fn, M); break;
							case 1: DHSModel(c, ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, rv, unit, Ft, Fn); break;
							}
							calculate_previous_rolling_resistance(cp->rfric, ir, jr, rc, Fn, Ft, res, tma);
							sumF += Fn + Ft;
							sumM += cross(rc, Fn + Ft);
							tsd[new_count] = sd;
							pair_id[new_count] = k;
							new_count++;
						}
						else if (cdist < 0 && cdist < coh_s)
						{
							double f = JKR_seperation_force(ir, jr, cp->coh);
							double cf = cohesionForce(ir, jr, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
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
	force[id] += sumF;
	moment[id] += sumM;
	if (new_count - sid > MAX_P2P_COUNT)
		printf("The total of contact with other particle is over(%d)\n.", new_count - sid);
	
	pair_count[id] = new_count - sid;
	tmax[id] = tma;
	rres[id] = res;
}

__device__ double particle_plane_contact_detection(
	device_plane_info& pe, double3& xp, double3& wp, double3& u, double r)
{
	double a_l1 = pow(wp.x - pe.l1, 2.0);
	double b_l2 = pow(wp.y - pe.l2, 2.0);
	double sqa = wp.x * wp.x;
	double sqb = wp.y * wp.y;
	double sqc = wp.z * wp.z;
	double sqr = r * r;

	if (abs(wp.z) < r && (wp.x > 0 && wp.x < pe.l1) && (wp.y > 0 && wp.y < pe.l2)) {
		double3 dp = xp - pe.xw;
		double3 uu = pe.uw / length(pe.uw);
		int pp = -sign(dot(dp, pe.uw));// dp.dot(pe.UW()));
		u = pp * uu;
		double collid_dist = r - abs(dot(dp, u));// dp.dot(u));
		return collid_dist;
	}

	if (wp.x < 0 && wp.y < 0 && (sqa + sqb + sqc) < sqr) {
		double3 Xsw = xp - pe.xw;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y < 0 && (a_l1 + sqb + sqc) < sqr) {
		double3 Xsw = xp - pe.w2;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x > pe.l1 && wp.y > pe.l2 && (a_l1 + b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe.w3;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	else if (wp.x < 0 && wp.y > pe.l2 && (sqa + b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe.w4;
		double h = length(Xsw);// .length();
		u = Xsw / h;
		return r - h;
	}
	if ((wp.x > 0 && wp.x < pe.l1) && wp.y < 0 && (sqb + sqc) < sqr) {
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w2 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.x < pe.l1) && wp.y > pe.l2 && (b_l2 + sqc) < sqr) {
		double3 Xsw = xp - pe.w4;
		double3 wj_wi = pe.w3 - pe.w4;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;

	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x < 0 && (sqr + sqc) < sqr) {
		double3 Xsw = xp - pe.xw;
		double3 wj_wi = pe.w4 - pe.xw;
		double3 us = wj_wi / length(wj_wi);// .length();
		double3 h_star = Xsw - (dot(Xsw, us)) * us;
		double h = length(h_star);// .length();
		u = -h_star / h;
		return r - h;
	}
	else if ((wp.x > 0 && wp.y < pe.l2) && wp.x > pe.l1 && (a_l1 + sqc) < sqr) {
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

__global__ void cluster_plane_contact_kernel(
	device_plane_info *plane,
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
	unsigned int new_count = sid;
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	for (unsigned int k = 0; k < cte.nplane; k++)
	{
		device_plane_info pl = plane[k];
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 unit = make_double3(0, 0, 0);
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));

		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, r);
		if (cdist > 0) {
			double2 sd = make_double2(0.0, 0.0);
			for (unsigned int i = 0; i < old_count; i++)
			{
				//unsigned int oid = i;
				if (p_pair_id[i] == k)
				{
					sd = p_tsd[i];
					break;
				}
			}
			double rcon = r - 0.5 * cdist;
			double3 cpt = ipos3 + r * unit;
			double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
			//double3 rc = r * unit;
			double3 dv = -(ivel + cross(iomega, dcpr));
			device_force_constant c = getConstant(
				1, r, 0.0, m, 0.0, cp->Ei, cp->Ej,
				cp->pri, cp->prj, cp->Gi, cp->Gj,
				cp->rest, cp->fric, cp->rfric, cp->sratio);
			switch (1)
			{
			case 0:
				HMCModel(
					c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist,
					iomega, dv, unit, Ft, Fn, M);
				break;
			case 1:
				DHSModel(
					c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist,
					iomega, sd.x, sd.y, dv, unit, Ft, Fn);
				break;
			}
			calculate_previous_rolling_resistance(
				cp->rfric, r, 0, dcpr, Fn, Ft, res, tma);
			sumF += Fn + Ft;
			sumM += cross(dcpr, Fn + Ft);
			//printf("po : %.16f, %.16f, %.16f\n", ipos3.x, ipos3.y, ipos3.z);
			//printf("dv : %.16f, %.16f, %.16f\n", dv.x, dv.y, dv.z);
			//printf("w : %.16f, %.16f, %.16f\n", iomega.x, iomega.y, iomega.z);
			//printf("dc : %.16f, %.16f, %.16f\n", dcpr.x, dcpr.y, dcpr.z);
			//printf("ev : %.16f, %.16f, %.16f, %.16f\n", ev[cid].x, ev[cid].y, ev[cid].z, ev[cid].w);
			/*printf("Fn : %.16f, %.16f, %.16f\n", Fn.x, Fn.y, Fn.z);
			printf("Ft : %.16f, %.16f, %.16f\n", Ft.x, Ft.y, Ft.z);*/
			tsd[new_count] = sd;
			pair_id[new_count] = k;
			new_count++;
		}
	}
	//printf("%f, %f, %f\n", sumF.x, sumF.y, sumF.z);
	force[id] += sumF;
	moment[id] += sumM;
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

template <int TCM>
__global__ void plane_contact_force_kernel(
	device_plane_info *plane,
	double4* pos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment,
	device_contact_property *cp, double* mass,
	double3* tmax, double* rres,
	unsigned int* pair_count, unsigned int* pair_id, double2* tsd, unsigned int np)
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
	unsigned int old_count = pair_count[id];
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
	unsigned int new_count = sid;
	double coh_s = 0.0;
	double res = 0.0;
	double3 tma = make_double3(0.0, 0.0, 0.0);
	for (unsigned int k = 0; k < cte.nplane; k++)
	{
		device_plane_info pl = plane[k];
		double3 dp = make_double3(ipos.x, ipos.y, ipos.z) - pl.xw;
		double3 unit = make_double3(0, 0, 0);
		double3 wp = make_double3(dot(dp, pl.u1), dot(dp, pl.u2), dot(dp, pl.uw));
		double cdist = particle_plane_contact_detection(pl, ipos3, wp, unit, r);
		if (cp->coh)
			coh_s = limit_cohesion_depth(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh);
		double2 sd = make_double2(0.0, 0.0);
		if (cdist > 0) {
			
			for (unsigned int i = 0; i < old_count; i++)
				if (p_pair_id[i] == k){ sd = p_tsd[i]; break; }
			
			double rcon = r - 0.5 * cdist;
			double3 rc = r * unit;
			double3 dv = -(ivel + cross(iomega, rc));
			device_force_constant c = getConstant(TCM, r, 0.0, m, 0.0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->Gi, cp->Gj, cp->rest, cp->fric, cp->rfric, cp->sratio);
			switch (TCM)
			{
			case 0: HMCModel(c, 0, 0, 0, 0, 0, 0, cp->coh, rcon, cdist, iomega, dv, unit, Ft, Fn, M); break;
			case 1: DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
			}
			calculate_previous_rolling_resistance(cp->rfric, r, 0, rc, Fn, Ft, res, tma);
			//printf("kn : %f, cn : %f, ks : %f, cs : %f", c.kn, c.vn, c.ks, c.vs);
			sumF += Fn + Ft;
			sumM += cross(rc, Fn + Ft);
			tsd[new_count] = sd;
			pair_id[new_count] = k;
			new_count++;
		}
		else if(cdist < 0 && cdist < coh_s)
		{
			double f = JKR_seperation_force(r, 0, cp->coh);
			double cf = cohesionForce(r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, f);
			sumF = sumF - cf * unit;
			tsd[new_count] = sd;
			pair_id[new_count] = k;
			new_count++;
		}
	}
	force[id] += sumF;
	moment[id] += sumM;
	if (new_count - sid > 3)
		printf("The total of contact with plane is over(%d)\n.", new_count - sid);
	pair_count[id] = new_count - sid;
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

__device__ float particle_cylinder_contact_detection(
	device_cylinder_info* cy, double4& pt, double3& u, double3& cp, unsigned int id = 0)
{
	double dist = -1.0;
	double3 ab = make_double3(cy->ptop.x - cy->pbase.x, cy->ptop.y - cy->pbase.y, cy->ptop.z - cy->pbase.z);
	double3 p = make_double3(pt.x, pt.y, pt.z);
	double t = dot(p - cy->pbase, ab) / dot(ab, ab);
	double3 _cp = make_double3(0.0, 0.0, 0.0);
	if (t >= 0 && t <= 1) {
		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		u = (_cp - p) / dist;
		cp = _cp - cy->len_rr.z * u;
		return cy->len_rr.y + pt.w - dist;
	}
	else {

		_cp = cy->pbase + t * ab;
		dist = length(p - _cp);
		if (dist < cy->len_rr.z) {
			double3 OtoCp = cy->origin - _cp;
			double OtoCp_ = length(OtoCp);
			u = OtoCp / OtoCp_;
			cp = _cp - cy->len_rr.z * u;
			return cy->len_rr.x * 0.5 + pt.w - OtoCp_;
		}
		double3 A_1 = makeTFM_1(cy->ep);
		double3 A_2 = makeTFM_2(cy->ep);
		double3 A_3 = makeTFM_3(cy->ep);
		double3 _at = p - cy->ptop;
		double3 at = toLocal(A_1, A_2, A_3, _at);
		//double r = length(at);
		cp = cy->ptop;
		if (abs(at.y) > cy->len_rr.x) {
			_at = p - cy->pbase;
			at = toLocal(A_1, A_2, A_3, _at);
			cp = cy->pbase;
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
		cp = cp + toGlobal(A_1, A_2, A_3, _cp);

		double3 disVec = cp - p;
		dist = length(disVec);
		u = disVec / dist;
		if (dist < pt.w) {
			return pt.w - dist;
		}
	}
	return -1.0;
}

template<int TCM>
__global__ void cylinder_hertzian_contact_force_kernel(
	device_cylinder_info *cy,
	double4* pos, double4 *ep, double3* vel, double4* ev,
	double3* force, double3* moment, device_contact_property *cp,
	double* mass, double3* mpos, double3* mf, double3* mm, unsigned int np)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= np)
		return;

	*mf = make_double3(0.0, 0.0, 0.0);
	*mm = make_double3(0.0, 0.0, 0.0);
	double cdist = 0.0;
	double im = mass[id];
	double4 ipos = make_double4(pos[id].x, pos[id].y, pos[id].z, pos[id].w);
	double3 ivel = make_double3(vel[id].x, vel[id].y, vel[id].z);
	double3 iomega = toAngularVelocity(ep[id], ev[id]);
	double3 unit = make_double3(0.0, 0.0, 0.0);
	double3 cpt = make_double3(0.0, 0.0, 0.0);
	double3 mp = make_double3(mpos->x, mpos->y, mpos->z);
	cdist = particle_cylinder_contact_detection(cy, ipos, unit, cpt, id);
	double3 si = cpt - mp;
	double3 cy2cp = cpt - cy->origin;
	double3 Ft = make_double3(0.0, 0.0, 0.0);
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 M = make_double3(0.0, 0.0, 0.0);
	double2 ds = make_double2(0.0, 0.0);
	double3 rc = make_double3(0, 0, 0);
	if (cdist > 0)
	{
		double rcon = ipos.w - 0.5 * cdist;
		rc = ipos.w * unit;
		double3 dv = cy->vel + cross(cy->omega, cy2cp) - (ivel + cross(iomega, ipos.w * unit));
		device_force_constant c = getConstant(
			TCM, ipos.w, 0, im, 0, cp->Ei, cp->Ej,
			cp->pri, cp->prj, cp->Gi, cp->Gj,
			cp->rest, cp->fric, cp->rfric, cp->sratio);
		switch (TCM)
		{
		case 0: HMCModel(
			c, 0, 0, 0, 0, 0, 0, 0, rcon, cdist, iomega,
			dv, unit, Ft, Fn, M);
			break;
		case 1:
			DHSModel(
				c, ipos.w, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, ds.x, ds.y,
				dv, unit, Ft, Fn);
			break;
		}
	}
	double3 sum_f = Fn + Ft;
	force[id] += sum_f;
	moment[id] += cross(rc, sum_f);// crossmake_double3(M.x, M.y, M.z);
	mf[id] = -(Fn);
	mm[id] = cross(si, -Fn);
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
	if ((blockSize >= 128) && (tid < 64)) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
	if ((blockSize >= 64) && (tid < 32)) { sdata[tid] = mySum = mySum + sdata[tid + 32]; } __syncthreads();
	if ((blockSize >= 32) && (tid < 16)) { sdata[tid] = mySum = mySum + sdata[tid + 16]; } __syncthreads();

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

	//if (ct == 2)
	//{
		if (d1 <= 0.0 && d2 <= 0.0) { ct = 0; return a; }
		if (d3 >= 0.0 && d4 <= d3) { ct = 0; return b; }
		if (d6 >= 0.0 && d5 <= d6) { ct = 0; return c; }
//	}
//	else if (ct == 1)
	//{
		if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { ct = 1; return a + (d1 / (d1 - d3)) * ab; }
		if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { ct = 1; return a + (d2 / (d2 - d6)) * ac; }
		if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { ct = 1; return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b); }
//	}	
	//else if (ct == 0)
	//{
		if (va >= 0 && vb >= 0 && vc >= 0)
		{
			double denom = 1.0 / (va + vb + vc);
			double3 v = vb * denom * ab;
			double3 w = vc * denom * ac;
			double3 _cpt = a + v + w;
			//double _dist = pr - length(p - _cpt);
			ct = 2;
			//if (_dist > 0) return _cpt;
			return _cpt; 
		}
	//}
	return make_double3(0, 0, 0);
}

__device__ bool checkConcave(device_triangle_info* dpi, unsigned int* tid, unsigned int kid, double4* dsph, unsigned int cnt)
{
	double3 p1 = make_double3(0, 0, 0);
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
	}
	return false;
}

__device__ bool checkOverlab(int3 ctype, double3 p, double3 c, double3 u0, double3 u1)
{
	bool b_over = false;
	if (p.x >= c.x - 1e-9 && p.x <= c.x + 1e-9) b_over = true;
	if (p.y >= c.y - 1e-9 && p.y <= c.y + 1e-9) b_over = true;
	if (p.z >= c.z - 1e-9 && p.z <= c.z + 1e-9) b_over = true;

	if (/*(ctype.y || ctype.z) &&*/ !b_over)
	{
		if (u0.x >= u1.x - 1e-9 && u0.x <= u1.x + 1e-9)
			if (u0.y >= u1.y - 1e-9 && u0.y <= u1.y + 1e-9)
				if (u0.z >= u1.z - 1e-9 && u0.z <= u1.z + 1e-9)
					b_over = true;
	}
	return b_over;
}

__device__ void particle_triangle_contact_force(
	unsigned int k,
	int t,
	device_triangle_info* dpi,
	device_contact_property *cp,
	device_mesh_mass_info* dpmi,
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
	unsigned int pidx = dpi[k].id;
	device_contact_property cmp = cp[pidx];
	device_mesh_mass_info pmi = dpmi[pidx];
	double3 cpt = closestPtPointTriangle(dpi[k], ipos, r, t);
	double3 po2cp = cpt - pmi.origin;
	double cdist = r - length(ipos - cpt);
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	device_triangle_info tri = dpi[k];
	double3 qp = tri.Q - tri.P;
	double3 rp = tri.R - tri.P;
//	double rcon = r - 0.5 * cdist;
	double3 unit = -cross(qp, rp);
	//double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
	//double3 dcpr_j = cpt - make_double3(jcpos.x, jcpos.y, jcpos.z);
	unit = unit / length(unit);
	double2 sd = make_double2(0.0, 0.0);
	for (unsigned int i = 0; i < old_count; i++)
		if (p_pair_id[i] == k){	sd = p_tsd[i]; break; }
	double3 rc = r * unit;
	double3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, rc));
	device_force_constant c = getConstant(1, r, 0, m, 0, cmp.Ei, cmp.Ej, cmp.pri, cmp.prj, cmp.Gi, cmp.Gj, cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
	switch (1)
	{
	case 1:	DHSModel( c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	}
	calculate_previous_rolling_resistance(cmp.rfric, r, 0, rc, Fn, Ft, res, tma);
	sum_force += Fn + Ft;
	sum_moment += cross(rc, Fn + Ft);
	dpmi[pidx].force += -(Fn + Ft);// +make_double3(1.0, 5.0, 9.0);
	dpmi[pidx].moment += -cross(po2cp, Fn + Ft);
	tsd[new_count] = sd;
	pair_id[new_count] = k;
	new_count++;
}

__device__ void cluster_triangle_contact_force(
	unsigned int k,
	int t,
	device_triangle_info* dpi,
	device_contact_property *cp,
	device_mesh_mass_info* dpmi,
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
	unsigned int pidx = dpi[k].id;
	device_contact_property cmp = cp[pidx];
	device_mesh_mass_info pmi = dpmi[pidx];
	double3 cpt = closestPtPointTriangle(dpi[k], ipos, r, t);
	double3 po2cp = cpt - pmi.origin;
	double cdist = r - length(ipos - cpt);
	double3 Fn = make_double3(0.0, 0.0, 0.0);
	double3 Ft = make_double3(0, 0, 0);
	device_triangle_info tri = dpi[k];
	double3 qp = tri.Q - tri.P;
	double3 rp = tri.R - tri.P;
//	double rcon = r - 0.5 * cdist;
	double3 unit = -cross(qp, rp);
	double3 dcpr = cpt - make_double3(icpos.x, icpos.y, icpos.z);
	//double3 dcpr_j = cpt - make_double3(jcpos.x, jcpos.y, jcpos.z);
	unit = unit / length(unit);
	double2 sd = make_double2(0.0, 0.0);
	for (unsigned int i = 0; i < old_count; i++)
		if (p_pair_id[i] == k) { sd = p_tsd[i]; break; }
	//double3 rc = r * unit;
	double3 dv = pmi.vel + cross(pmi.omega, po2cp) - (ivel + cross(iomega, dcpr));
	device_force_constant c = getConstant(1, r, 0, m, 0, cmp.Ei, cmp.Ej, cmp.pri, cmp.prj, cmp.Gi, cmp.Gj, cmp.rest, cmp.fric, cmp.rfric, cmp.sratio);
	switch (1)
	{
	case 1:	DHSModel(c, r, 0, cp->Ei, cp->Ej, cp->pri, cp->prj, cp->coh, cdist, iomega, sd.x, sd.y, dv, unit, Ft, Fn); break;
	}
	calculate_previous_rolling_resistance(cmp.rfric, r, 0, dcpr, Fn, Ft, res, tma);
	sum_force += Fn + Ft;
	sum_moment += cross(dcpr, Fn + Ft);
	dpmi[pidx].force += -(Fn + Ft);// +make_double3(1.0, 5.0, 9.0);
	dpmi[pidx].moment += -cross(po2cp, Fn + Ft);
	tsd[new_count] = sd;
	pair_id[new_count] = k;
	new_count++;
}

template<int TCM>
__global__ void particle_polygonObject_collision_kernel(
	device_triangle_info* dpi, device_mesh_mass_info* dpmi,
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
	unsigned int new_count = sid;

	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	int3 ctype = make_int3(0, 0, 0);
	unsigned int ctriangle[5] = { 0, };
	unsigned int cline[5] = { 0, };
	unsigned int cpoint[5] = { 0, };
	unsigned int nct = 0;
	unsigned int ncl = 0;
	unsigned int ncp = 0;
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
						if (k >= cte.np)
						{
							k -= cte.np;
							int t = -1;
							//unsigned int pidx = dpi[k].id;
							//device_contact_property cmp = cp[pidx];
							//device_mesh_mass_info pmi = dpmi[pidx];
							//double3 po2cp = cpt - pmi.origin;
							double cdist = ir - length(ipos - closestPtPointTriangle(dpi[k], ipos, ir, t));
							if (cdist > 0)
							{
								switch (t)
								{
								case 2: ctriangle[nct++] = k; break;
								case 1: cline[ncl++] = k; break;
								case 0: cpoint[ncp++] = k; break;
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
	printf("tlp : [%d - %d - %d]\n", nct, ncl, ncp);
	for (unsigned int k = 0; k < nct; k++)
		particle_triangle_contact_force(ctriangle[k], 0, dpi, cp, dpmi, ipos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if(!nct)
		for(unsigned int k = 0; k < ncl; k++)
			particle_triangle_contact_force(cline[k], 1, dpi, cp, dpmi, ipos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if(!nct && !ncl)
		for (unsigned int k = 0; k < ncp; k++)
			particle_triangle_contact_force(cpoint[k], 2, dpi, cp, dpmi, ipos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);

	force[id] += sum_force;
	moment[id] += sum_moment;
	///*if (new_count - sid > MAX_P2MS_COUNT)
	//	printf("The total of contact with triangle */is over(%d)\n.", new_count - sid);
	pair_count[id] = new_count - sid;
	tmax[id] += tma;
	rres[id] += res;
}

__global__ void cluster_meshes_contact_kernel(
	device_triangle_info *dpi, device_mesh_mass_info* dpmi,
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
	unsigned int new_count = sid;

	double3 previous_cpt = make_double3(0.0, 0.0, 0.0);
	double3 previous_unit = make_double3(0.0, 0.0, 0.0);
	unsigned int start_index = 0;
	unsigned int end_index = 0;
	int3 ctype = make_int3(0, 0, 0);
	unsigned int ctriangle[5] = { 0, };
	unsigned int cline[5] = { 0, };
	unsigned int cpoint[5] = { 0, };
	unsigned int nct = 0;
	unsigned int ncl = 0;
	unsigned int ncp = 0;
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
						if (k >= cte.np)
						{
							k -= cte.np;
							int t = -1;
							//unsigned int pidx = dpi[k].id;
							//device_contact_property cmp = cp[pidx];
							//d//evice_mesh_mass_info pmi = dpmi[pidx];
							double cdist = ir - length(ipos - closestPtPointTriangle(dpi[k], ipos, ir, t));
							if (cdist > 0)
							{
								switch (t)
								{
								case 2: ctriangle[nct++] = k; break;
								case 1: cline[ncl++] = k; break;
								case 0: cpoint[ncp++] = k; break;
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
	printf("tlp : [%d - %d - %d]\n", nct, ncl, ncp);
	for (unsigned int k = 0; k < nct; k++)
		cluster_triangle_contact_force(ctriangle[k], 0, dpi, cp, dpmi, ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if (!nct)
		for (unsigned int k = 0; k < ncl; k++)
			cluster_triangle_contact_force(cline[k], 1, dpi, cp, dpmi, ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);
	if (!nct && !ncl)
		for (unsigned int k = 0; k < ncp; k++)
			cluster_triangle_contact_force(cpoint[k], 2, dpi, cp, dpmi, ipos, icpos, ivel, iomega, old_count, p_pair_id, p_tsd, pair_id, tsd, ir, im, sum_force, sum_moment, res, tma, new_count);

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
		if (length(_Tr) >= length(_Tmax))
			_Tr = _Tmax;
		moment[id] += _Tr;
	}
	//rfm[id] = make_double4(0.0, 0.0, 0.0, 0.0);
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
	device_mesh_mass_info *dpmi, double4* mep, double* vList,
	double4* sphere, double3* dlocal, device_triangle_info* dpi, unsigned int ntriangle)
{
	unsigned id = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int np = ntriangle;
	if (id >= np)
		return;
	int s = id * 9;
	int mid = dpi[id].id;
//	printf("idx(%d) : mid = %d\n", id, mid);
	double3 pos = dpmi[mid].origin;
	double4 ep = mep[mid];// dpmi[mid].ep;
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