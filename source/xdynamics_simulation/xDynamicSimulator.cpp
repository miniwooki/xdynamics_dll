#include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_simulation/xIntegratorHHT.h"
#include "xdynamics_simulation/xIntegratorRK4.h"
#include "xdynamics_simulation/xIngegratorVV.h"
#include "xdynamics_simulation/xIncompressibleSPH.h"
#include "xdynamics_simulation/xKinematicAnalysis.h"
#include "xdynamics_global.h"
#include <chrono>

typedef xUtilityFunctions xuf;

xDynamicsSimulator::xDynamicsSimulator()
	: xSimulation()
	, xdm(NULL)
	, xmbd(NULL)
	, xdem(NULL)
	, xsph(NULL)
	//, stop_condition(NULL)
{

}

xDynamicsSimulator::xDynamicsSimulator(xDynamicsManager* _xdm)
	: xSimulation()
	, xdm(_xdm)
	, xmbd(NULL)
	, xdem(NULL)
	, xsph(NULL)
	//, stop_condition(NULL)
{
// 	if (xdm->XMBDModel())
// 	{
// 		xmbd = new xMultiBodySimulation;
// 		xmbd->initialize(xdm->XMBDModel());
// 	}
	//SET_GLOBAL_XDYNAMICS_MANAGER(_xdm);
	//SET_GLOBAL_XDYNAMICS_SIMULATOR(this);
}

xDynamicsSimulator::~xDynamicsSimulator()
{
	if (xmbd) delete xmbd; xmbd = NULL;
	if (xdem) delete xdem; xdem = NULL;
	//xdm->release_result_manager();
	//if (stop_condition) delete stop_condition; stop_condition = NULL;
}

xMultiBodySimulation* xDynamicsSimulator::setupMBDSimulation(xSimulation::MBDSolverType mst)
{
	xSimulation::mbd_solver_type = mst;
	switch (mst)
	{
	case xSimulation::IMPLICIT_HHT:
		if (!xmbd)
			xmbd = new xIntegratorHHT;
	default:
		break;
	}
	return xmbd;
}

bool xDynamicsSimulator::xInitialize(
	bool exefromgui, double _dt, unsigned int _st, double _et, unsigned int _sp,
	xMultiBodySimulation* _xmbd, 
	xDiscreteElementMethodSimulation* _xdem,
	xSmoothedParticleHydrodynamicsSimulation* _xsph)
{
	std::string c_path = xModel::makeFilePath(xModel::getModelName() + ".cdn");
	std::fstream fs;
	fs.open(c_path, std::ios::out | std::ios::binary);
	if (!fs.is_open())
	{
		return false;
	}
 	if(_dt) xSimulation::dt = _dt;
 	if(_st) xSimulation::st = _st;
 	if(_et) xSimulation::et = _et;
	xSimulation::nstep = static_cast<unsigned int>((xSimulation::et / xSimulation::dt));
	xSimulation::npart = static_cast<unsigned int>((nstep / xSimulation::st)) + 1;
	//fs.write((char*)&xSimulation::nstep, sizeof(unsigned int));
	fs.write((char*)&xSimulation::npart, sizeof(unsigned int));                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
	if(!_sp)
		xdm->initialize_result_manager(xSimulation::npart);
	if (_xmbd)
	{
		xmbd = _xmbd;
	}
	else
	{
		switch (xSimulation::mbd_solver_type)
		{
		case xSimulation::EXPLICIT_RK4: xmbd = new xIntegratorRK4(); break;
		case xSimulation::IMPLICIT_HHT: xmbd = new xIntegratorHHT(); break;
		case xSimulation::KINEMATIC: xmbd = new xKinematicAnalysis(); break;
		}
	}
	if (_xdem) xdem = _xdem;
	else
	{
		switch (xSimulation::dem_solver_type)
		{
		case xSimulation::EXPLICIT_VV: xdem = new xIntegratorVV(); break;
		}
	}
	if (_xsph) xsph = _xsph;
	else
	{
		switch (xSimulation::sph_solver_type)
		{
		case xSimulation::INCOMPRESSIBLESPH: xsph = new xIncompressibleSPH(); break;
		case xSimulation::WEAKELYSPH: break;
		}
	}
	if (xmbd)
	{

		//std::cout << xmbd->Initialized() << " " << xdm->XMBDModel() << std::endl;
		if (xdm->XMBDModel() && !xmbd->Initialized())
		{
			xLog::log("An uninitialized multibody model has been detected.");
			//int ret = xmbd->Initialize(xdm->XMBDModel());
			if (!checkXerror(xmbd->Initialize(xdm->XMBDModel(), !_sp)))
			{
				char id_mbd = 'm';
				unsigned int mdim = xmbd->num_generalized_coordinate();
				unsigned int sdim = xmbd->num_constraint_equations();
				fs.write(&id_mbd, sizeof(char));
				fs.write((char*)&mdim, sizeof(unsigned int));
				fs.write((char*)&sdim, sizeof(unsigned int));
				xLog::log("The initialization of multibody model was succeeded.");
			}				
		}
	}
	if (xdem)
	{
		if (xdm->XDEMModel() && !xdem->Initialized())
		{
			xLog::log("An uninitialized discrete element method model has been detected.");
			//int ret = xdem->Initialize(xdm->XDEMModel(), xdm->XContact());
			if (!checkXerror(xdem->Initialize(xdm->XDEMModel(), xdm->XContact(), !_sp)))
			{
				char id_dem = 'd';
				unsigned int nparticle = xdem->num_particles();
				unsigned int ncluster = xdem->num_clusters();
				fs.write(&id_dem, sizeof(char));
				fs.write((char*)&nparticle, sizeof(unsigned int));
				fs.write((char*)&ncluster, sizeof(unsigned int));
				xLog::log("The initialization of discrete element method model was succeeded.");
			}
				
		}
		if (xmbd)
		{
			xdem->updateObjectFromMBD();
		}
		//xdem->EnableSaveResultToMemory(exefromgui);
	}
	if (xsph)
	{
		if (xdm->XSPHModel() && !xsph->Initialized())
		{
			xLog::log("An uninitialized smoothed particle hydrodynamics model has been detected.");
			if (!checkXerror(xsph->Initialize(xdm->XSPHModel())))
				xLog::log("The initialization of smoothed particle hydrodynamics mode was succeeded.");
		}
	}
	if(!_sp)
		savePartData(0, 0);
	fs.close();
	xdm->XResult()->set_gpu_process_device(xSimulation::Gpu());
	return true;
}

//bool xDynamicsSimulator::xInitialize_from_part_result(bool exefromgui, double _dt, unsigned int _st, double _et, xMultiBodySimulation * _xmbd, xDiscreteElementMethodSimulation * _xdem, xSmoothedParticleHydrodynamicsSimulation * _xsph)
//{
//	std::string c_path = xModel::makeFilePath(xModel::getModelName() + ".cdn");
//	std::fstream fs;
//	fs.open(c_path, std::ios::out | std::ios::binary);
//	if (!fs.is_open())
//	{
//		return false;
//	}
//	if (_dt) xSimulation::dt = _dt;
//	if (_st) xSimulation::st = _st;
//	if (_et) xSimulation::et = _et;
//	xSimulation::nstep = static_cast<unsigned int>((xSimulation::et / xSimulation::dt));
//	xSimulation::npart = static_cast<unsigned int>((nstep / xSimulation::st)) + 1;
//	//fs.write((char*)&xSimulation::nstep, sizeof(unsigned int));
//	fs.write((char*)&xSimulation::npart, sizeof(unsigned int));
//	xdm->initialize_result_manager(xSimulation::npart);
//	if (_xmbd)
//	{
//		xmbd = _xmbd;
//	}
//	else
//	{
//		switch (xSimulation::mbd_solver_type)
//		{
//		case xSimulation::EXPLICIT_RK4: xmbd = new xIntegratorRK4(); break;
//		case xSimulation::IMPLICIT_HHT: xmbd = new xIntegratorHHT(); break;
//		case xSimulation::KINEMATIC: xmbd = new xKinematicAnalysis(); break;
//		}
//	}
//	if (_xdem) xdem = _xdem;
//	else
//	{
//		switch (xSimulation::dem_solver_type)
//		{
//		case xSimulation::EXPLICIT_VV: xdem = new xIntegratorVV(); break;
//		}
//	}
//	if (_xsph) xsph = _xsph;
//	else
//	{
//		switch (xSimulation::sph_solver_type)
//		{
//		case xSimulation::INCOMPRESSIBLESPH: xsph = new xIncompressibleSPH(); break;
//		case xSimulation::WEAKELYSPH: break;
//		}
//	}
//	if (xmbd)
//	{
//
//		//std::cout << xmbd->Initialized() << " " << xdm->XMBDModel() << std::endl;
//		if (xdm->XMBDModel() && !xmbd->Initialized())
//		{
//			xLog::log("An uninitialized multibody model has been detected.");
//			//int ret = xmbd->Initialize(xdm->XMBDModel());
//			if (!checkXerror(xmbd->Initialize(xdm->XMBDModel())))
//			{
//				char id_mbd = 'm';
//				unsigned int mdim = xmbd->num_generalized_coordinate();
//				unsigned int sdim = xmbd->num_constraint_equations();
//				fs.write(&id_mbd, sizeof(char));
//				fs.write((char*)&mdim, sizeof(unsigned int));
//				fs.write((char*)&sdim, sizeof(unsigned int));
//				xLog::log("The initialization of multibody model was succeeded.");
//			}
//		}
//	}
//	if (xdem)
//	{
//		if (xdm->XDEMModel() && !xdem->Initialized())
//		{
//			xLog::log("An uninitialized discrete element method model has been detected.");
//			//int ret = xdem->Initialize(xdm->XDEMModel(), xdm->XContact());
//			if (!checkXerror(xdem->Initialize(xdm->XDEMModel(), xdm->XContact())))
//			{
//				char id_dem = 'd';
//				unsigned int nparticle = xdem->num_particles();
//				unsigned int ncluster = xdem->num_clusters();
//				fs.write(&id_dem, sizeof(char));
//				fs.write((char*)&nparticle, sizeof(unsigned int));
//				fs.write((char*)&ncluster, sizeof(unsigned int));
//				xLog::log("The initialization of discrete element method model was succeeded.");
//			}
//
//		}
//		if (xmbd)
//		{
//			xdem->updateObjectFromMBD();
//		}
//		//xdem->EnableSaveResultToMemory(exefromgui);
//	}
//	if (xsph)
//	{
//		if (xdm->XSPHModel() && !xsph->Initialized())
//		{
//			xLog::log("An uninitialized smoothed particle hydrodynamics model has been detected.");
//			if (!checkXerror(xsph->Initialize(xdm->XSPHModel())))
//				xLog::log("The initialization of smoothed particle hydrodynamics mode was succeeded.");
//		}
//	}
//	savePartData(0, 0);
//	fs.close();
//	return true;
//}

bool xDynamicsSimulator::savePartData(double ct, unsigned int pt)
{
	if (xdem)
	{
		xdem->SaveStepResult(pt);
		xdm->XContact()->SaveStepResult(pt, xdem->num_particles());
	}
	if (xmbd)
	{
		xmbd->SaveStepResult(pt);
	}
	if (xsph)
		xsph->SaveStepResult(pt, ct);
	xdm->XResult()->export_step_data_to_file(pt, ct);
	xdm->XResult()->set_current_part_number(pt);
	return true;
}

void xDynamicsSimulator::exportPartData()
{
 	std::fstream of;
 	of.open((xModel::path + xModel::name + "/" + xModel::name + ".rlt").toStdString(), std::ios::out);
	of << "SIMULATION" << endl;
	of << xSimulation::dt << " " << xSimulation::st << " " << xSimulation::et << " " << xSimulation::npart << std::endl;
	if (xmbd)
	{
		of << "MBD" << endl;
		xmbd->ExportResults(of);
	}
	
	if (xdem)
	{
		of << "DEM" << endl;
		xdem->ExportResults(of);
	}
		
	of.close();
}

unsigned int xDynamicsSimulator::setupByLastSimulationFile(std::string lmr, std::string ldr)
{
	unsigned int pt = 0;
	if (!lmr.empty())
	{
		pt = xmbd->setupByLastSimulationFile(lmr);
	}
	if (!ldr.empty())
	{
		pt = xdem->setupByLastSimulationFile(ldr);
	}
	return pt;
}

bool xDynamicsSimulator::checkStopCondition()
{
	xmap<xstring, xObject*>* _xobjs = &xdm->XObject()->XObjects();
	for (xmap<xstring, xObject*>::iterator it = _xobjs->begin(); it != _xobjs->end(); it.next())// (xObject* xo, xdm->XObject()->XObjects())
	{
		xPointMass* xpm = dynamic_cast<xPointMass*>(it.value());
		if (xpm)
		{
			if (xpm->checkStopCondition())
				return true;
		}		
	}
	return false;
}

bool xDynamicsSimulator::xRunSimulationThread(double ct, unsigned int cstep)
{
	try
	{
		if (xsph)
			if (checkXerror(xsph->OneStepSimulation(ct, cstep)))
				return false;
		if (xdem)
		{
			if (!xmbd)
			{
				xdm->XObject()->UpdateMovingObjects(xSimulation::dt);
				xdem->updateObjectFromMBD();
			}
			else
			{
				xmbd->SetZeroBodyForce();
			}
			if (checkXerror(xdem->OneStepSimulation(ct, cstep)))
				return false;
		}
		if (xmbd)
		{
			if (checkXerror(xmbd->OneStepSimulation(ct, cstep)))
				throw runtime_error(xDynamicsError::getErrorString());
			if (xdem)
				xdem->updateObjectFromMBD();
			//xmbd->SetZeroBodyForce();
		}
	}
	catch (std::bad_alloc &e)
	{

	}
	catch (std::exception &e)
	{
		xLog::log("Exception in dynamic simulator : " + std::string(e.what()));
		return false;
	}
	if (checkStopCondition())
		xSimulation::triggerStopSimulation();
//	return xDynamicsError::xdynamicsErrorDiscreteElementMethodModelInitialization;
	return true;
}

double xDynamicsSimulator::set_from_part_result(std::string path)
{
	std::fstream fs;
	fs.open(path, std::ios::in | std::ios::binary);
	unsigned int cnt = 0;
	unsigned int np = 0;
	unsigned int ns = 0;
	double ct = 0;
	
	fs.read((char*)&ct, sizeof(double));
	if (xdem)
	{
		//double* _cpos = NULL, *_pos = NULL, *_vel = NULL, *_acc = NULL, *_ep = NULL, *_ev = NULL, *_ea = NULL;
		np = xdem->num_particles();
		ns = xdem->num_clusters();
		double* _pos = new double[np * 4];
		double* _vel = new double[ns * 3];
		double* _acc = new double[ns * 3];
		double* _ep = new double[ns * 4];
		double* _ev = new double[ns * 4];
		double* _ea = new double[ns * 4];
		double* _cpos = NULL;
		if (ns != np)
			_cpos = new double[ns * 4];
		
	//	xrm->get_times()[cnt] = ct;
		//time[cnt] = ct;
		if (np)
		{
			unsigned int _np = 0;
			unsigned int _ns = 0;
			fs.read((char*)&_np, sizeof(unsigned int));
			fs.read((char*)&_ns, sizeof(unsigned int));
			fs.read((char*)_pos, sizeof(double) * np * 4);
			fs.read((char*)_vel, sizeof(double) * ns * 3);
			fs.read((char*)_acc, sizeof(double) * ns * 3);
			fs.read((char*)_ep, sizeof(double) * ns * 4);
			fs.read((char*)_ev, sizeof(double) * ns * 4);
			fs.read((char*)_ea, sizeof(double) * ns * 4);
			if (ns != np)
				fs.read((char*)_cpos, sizeof(double) * ns * 4);
			xdem->set_dem_data(_cpos, _pos, _vel, _acc, _ep, _ev, _ea);
		}
		delete[] _pos;
		delete[] _vel;
		delete[] _acc;
		delete[] _ep;
		delete[] _ev;
		delete[] _ea;
		if(_cpos) 
			delete[] _cpos;
	}
	if (xmbd)
	{
		unsigned int m_size = 0;
		unsigned int j_size = 0;
		unsigned int d_size = 0;
		fs.read((char*)&m_size, sizeof(unsigned int));
		fs.read((char*)&j_size, sizeof(unsigned int));
		fs.read((char*)&d_size, sizeof(unsigned int));
		//xMultiBodyModel* xmbd = xdm->XMBDModel();
		for (unsigned int i = 0; i < m_size; i++)
		{
			xPointMass::pointmass_result pr = { 0, };
			fs.read((char*)&pr, sizeof(xPointMass::pointmass_result));
		}
		for (unsigned int i = 0; i < j_size; i++)
		{
			xKinematicConstraint::kinematicConstraint_result kr = { 0, };
			fs.read((char*)&kr, sizeof(xKinematicConstraint::kinematicConstraint_result));
		}
		for (unsigned int i = 0; i < d_size; i++)
		{
			xDrivingRotationResultData xdr = { 0, };
			fs.read((char*)&xdr, sizeof(xDrivingRotationResultData));
			xmbd->Model()->set_driving_rotation_data(i, xdr);
		}
		unsigned int ng = xmbd->num_generalized_coordinate() + xModel::OneDOF();
		unsigned int nt = xmbd->num_generalized_coordinate() + xmbd->num_constraint_equations();
		double *q = new double[ng];
		double *dq = new double[ng];
		double *q_1 = new double[ng];
		double *rhs = new double[nt];

		fs.read((char*)q, sizeof(double) * ng);
		fs.read((char*)dq, sizeof(double) * ng);
		fs.read((char*)q_1, sizeof(double) * ng);
		fs.read((char*)rhs, sizeof(double) * nt);
		xmbd->set_mbd_data(q, dq, q_1, rhs);
		delete[] q;
		delete[] dq;
		delete[] q_1;
		delete[] rhs;
	}
	if (xdm->XContact())
	{
		xdm->XContact()->set_from_part_result(fs);
	}
	fs.close();
	return ct;
}

bool xDynamicsSimulator::xRunSimulation()
{
// 	isStop = false;
	char buf[255] = { 0, };
 	unsigned int part = 0;
 	unsigned int cstep = 0;
 	unsigned int eachStep = 0;
	unsigned int numPart = 0;
 	double dt = xSimulation::dt;
 	double ct = dt * cstep;
// 	bool isUpdated = false;
 	xSimulation::setCurrentTime(ct);
 	double total_time = 0.0;
	double elapsed_time = 0.0;
	double previous_time = 0.0;
	xLog::log("=========  =======    ==========    ======   ========   =============  ====================");
	xLog::log("PART       SimTime    TotalSteps    Steps    Time/Sec   TotalTime/Sec  Finish time         ");
	xLog::log("=========  =======    ==========    ======   ========   =============  ====================");
	//xTime tme;
	/*QTime tme;
	QTime startTime = tme.currentTime();
	QDate startDate = QDate::currentDate();
	tme.start();*/
	chrono::system_clock::time_point start = chrono::system_clock::now();
	while (cstep < nstep)
	{
		cstep++;
		eachStep++;
//		std::cout << cstep << std::endl;
		ct += xSimulation::dt;
		xSimulation::setCurrentTime(ct);
		if (xsph)
			if (checkXerror(xsph->OneStepSimulation(ct, cstep)))
				return false;
		if (xdem)
			if (checkXerror(xdem->OneStepSimulation(ct, cstep)))
				return false;
		if (xmbd)
		{
			if (checkXerror(xmbd->OneStepSimulation(ct, cstep)))
				return false;
			if (xdem)
				xdem->updateObjectFromMBD();
		}
			
		if (!((cstep) % xSimulation::st))
		{
			previous_time = elapsed_time;
			chrono::system_clock::time_point end = chrono::system_clock::now();
			chrono::duration<double> sec = end - start;
			elapsed_time = sec.count();// tme.elapsed() * 0.001;
			total_time += elapsed_time - previous_time;
			part++;
			if (savePartData(ct, part))
			{
				sprintf_s(buf, "Part%04d   %4.5f %10d      %5d      %4.5f     %4.5f    %s", part, ctime, cstep, eachStep, elapsed_time - previous_time, total_time, xUtilityFunctions::GetDateTimeFormat("%d-%m-%y %H:%M:%S", 0).c_str());
				std::cout << buf << std::endl;
				xLog::log(buf);
			}
			eachStep = 0;
		}
	}
	chrono::system_clock::time_point end = chrono::system_clock::now();
	chrono::duration<double> sec = end - start;
	elapsed_time = sec.count();// tme.elapsed() * 0.001;
	total_time += elapsed_time;
	xLog::log("=========  =======    ==========    ======   ========   =============  ====================\n");
//	exportPartData();
	/*QTime endTime = QTime::currentTime();
	QDate endDate = QDate::currentDate();
	int minute = static_cast<int>(total_time / 60.0);
	int hour = static_cast<int>(minute / 60.0);
	double second = total_time - (hour * 3600 + minute * 60);
	xLog::log(
		"     Starting time/date     = " + startTime.toString().toStdString() + " / " + startDate.toString().toStdString() + "\n" +
		"     Ending time/date       = " + endTime.toString().toStdString() + " / " + endDate.toString().toStdString() + "\n" +
		"     CPU + GPU time         = " + QString("%1").arg(total_time).toStdString() + " second  ( " + QString("%1").arg(hour).toStdString() + " h. " + QString("%1").arg(minute).toStdString() + " m. " + QString("%1").arg(second).toStdString() + " s. )");*/
	return true;
}