#include "xdynamics_algebra/xParaboalPredictor.h"
#include "xdynamics_algebra/xAlgebraMath.h"

xParabolaPredictor::xParabolaPredictor()
	: data3(NULL)
{

}

xParabolaPredictor::~xParabolaPredictor()
{
	if (data3) delete data3; data3 = NULL;
}

void xParabolaPredictor::init(double* _data, int _dataSize)
{
	data = _data;
	dataSize = _dataSize;
	data3 = new xVectorD(dataSize * 3);
}

bool xParabolaPredictor::apply(unsigned int it)
{
	int insertID = (it - 1) % 3;
	for (int i(0); i < dataSize; i++) (*data3)(insertID * dataSize + i) = *(data + i);
	if (it < 3) return false;
	double cur_xp = dt * it;
	xp = new_vector3d((it - 3) * dt, (it - 2) * dt, (it - 1) * dt);
	switch (insertID)
	{
	case 2: idx = new_vector3i(0, 1, 2); break;
	case 0: idx = new_vector3i(1, 2, 0); break;
	case 1: idx = new_vector3i(2, 0, 1); break;
	}
	A = { xp.x * xp.x, xp.x, 1
		, xp.y * xp.y, xp.y, 1
		, xp.z * xp.z, xp.z, 1 };

	inverse(A);
	//fstream of;
	//of.open("C:/predictor_data.txt", ios::out);

	for (int i(0); i < dataSize; i++)
	{
		yp = new_vector3d((*data3)(idx.x * dataSize + i), (*data3)(idx.y * dataSize + i), (*data3)(idx.z * dataSize + i));
		//std::cout << yp << std::endl;
		coeff = A * yp;

		data[i] = coeff.x * cur_xp * cur_xp + coeff.y * cur_xp + coeff.z;
	}
	//of.close();

	return true;
}