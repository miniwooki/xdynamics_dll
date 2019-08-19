#include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_simulation/xIntegratorHHT.h"
#include "xdynamics_simulation/xIntegratorRK4.h"
#include "xdynamics_simulation/xIngegratorVV.h"
#include "xdynamics_simulation/xIncompressibleSPH.h"
#include "xdynamics_simulation/xKinematicAnalysis.h"
#include "xdynamics_global.h"
#include <QtCore/QTime>
#include <QtCore/QDate>

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
	SET_GLOBAL_XDYNAMICS_MANAGER(_xdm);
	SET_GLOBAL_XDYNAMICS_SIMULATOR(this);
}

xDynamicsSimulator::~xDynamicsSimulator()
{
	if (xmbd) delete xmbd; xmbd = NULL;
	if (xdem) delete xdem; xdem = NULL;
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
	bool exefromgui, double _dt, unsigned int _st, double _et, 
	xMultiBodySimulation* _xmbd, 
	xDiscreteElementMethodSimulation* _xdem,
	xSmoothedParticleHydrodynamicsSimulation* _xsph)
{
 	if(_dt) xSimulation::dt = _dt;
 	if(_st) xSimulation::st = _st;
 	if(_et) xSimulation::et = _et;
	xSimulation::nstep = static_cast<unsigned int>((xSimulation::et / xSimulation::dt));
	xSimulation::npart = static_cast<unsigned int>((nstep / xSimulation::st)) + 1;
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
			if (!checkXerror(xmbd->Initialize(xdm->XMBDModel())))
				xLog::log("The initialization of multibody model was succeeded.");
		}
	}
	if (xdem)
	{
		if (xdm->XDEMModel() && !xdem->Initialized())
		{
			xLog::log("An uninitialized discrete element method model has been detected.");
			//int ret = xdem->Initialize(xdm->XDEMModel(), xdm->XContact());
			if (!checkXerror(xdem->Initialize(xdm->XDEMModel(), xdm->XContact())))
				xLog::log("The initialization of discrete element method model was succeeded.");
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
	//if (xdem && xmbd)
	//{
	//	xSpringDamperForce *xsdf = xdm->XDEMModel()->XSpringDamperForce();
	//	if (xsdf)
	//	{
	//		//xsdf->SetDEMParticlePosition(xdem->HostPosition(), xdem->HostVelocity());
	//		xmbd->SetDEMSpringDamper(xsdf);
	//	}
	//	xmbd->setDEMPositionVelocity(xdem->Position(), xdem->Velocity());
	//}
// 	xuf::DeleteFileByEXT(xuf::xstring(xModel::path) + xuf::xstring(xModel::name), "bin");
// 	xuf::DeleteFileByEXT(xuf::xstring(xModel::path) + xuf::xstring(xModel::name), "bpm");
// 	xuf::DeleteFileByEXT(xuf::xstring(xModel::path) + xuf::xstring(xModel::name), "bkc");
	savePartData(0, 0);
	return true;
}

bool xDynamicsSimulator::savePartData(double ct, unsigned int pt)
{
	if (xmbd)
		xmbd->SaveStepResult(pt, ct);
	if (xdem)
	{
		xdem->SaveStepResult(pt, ct);
		if (!xmbd)
		{
			xdm->XObject()->SaveResultCompulsionMovingObjects(ct);
		}
	}		
	if (xsph)
		xsph->SaveStepResult(pt, ct);
	return true;
}

void xDynamicsSimulator::exportPartData()
{
// 	QFile qf(xModel::path + xModel::name + "/result_list.rlt");
// 	qf.open(QIODevice::WriteOnly);
// 	QTextStream qts(&qf);
 	std::fstream of;
 	of.open((xModel::path + xModel::name + "/" + xModel::name + ".rlt").toStdString(), std::ios::out);
	
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

bool xDynamicsSimulator::checkStopCondition()
{
	foreach(xObject* xo, xdm->XObject()->XObjects())
	{
		xPointMass* xpm = dynamic_cast<xPointMass*>(xo);
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
			
		if (checkXerror(xdem->OneStepSimulation(ct, cstep)))
			return false;
	}
	if (xmbd)
	{
		if (checkXerror(xmbd->OneStepSimulation(ct, cstep)))
			return false;
		if (xdem)
			xdem->updateObjectFromMBD();
		xmbd->SetZeroBodyForce();
	}
	if (checkStopCondition())
		xSimulation::triggerStopSimulation();
//	return xDynamicsError::xdynamicsErrorDiscreteElementMethodModelInitialization;
	return true;
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
	QTime tme;
	QTime startTime = tme.currentTime();
	QDate startDate = QDate::currentDate();
	tme.start();
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
			elapsed_time = tme.elapsed() * 0.001;
			total_time += elapsed_time - previous_time;
			part++;
			if (savePartData(ct, part))
			{
				sprintf_s(buf, "Part%04d   %4.5f %10d      %5d      %4.5f     %4.5f    %s", part, ctime, cstep, eachStep, elapsed_time - previous_time, total_time, xUtilityFunctions::GetDateTimeFormat("%d-%m-%y %H:%M:%S", 0).c_str());
				xLog::log(buf);
			}
			eachStep = 0;
		}
	}
	//tme.stop();
	elapsed_time = tme.elapsed() * 0.001;
	total_time += elapsed_time;
	xLog::log("=========  =======    ==========    ======   ========   =============  ====================\n");
	exportPartData();
	QTime endTime = QTime::currentTime();
	QDate endDate = QDate::currentDate();
	int minute = static_cast<int>(total_time / 60.0);
	int hour = static_cast<int>(minute / 60.0);
	double second = total_time - (hour * 3600 + minute * 60);
	xLog::log(
		"     Starting time/date     = " + startTime.toString().toStdString() + " / " + startDate.toString().toStdString() + "\n" +
		"     Ending time/date       = " + endTime.toString().toStdString() + " / " + endDate.toString().toStdString() + "\n" +
		"     CPU + GPU time         = " + QString("%1").arg(total_time).toStdString() + " second  ( " + QString("%1").arg(hour).toStdString() + " h. " + QString("%1").arg(minute).toStdString() + " m. " + QString("%1").arg(second).toStdString() + " s. )");
	return true;
// 		QMutexLocker locker(&m_mutex);
// 		if (isStop)
// 			break;
// 		cstep++;
// 		eachStep++;
// 		ct += simulation::dt;
// 		qDebug() << ct;
// 
// 		simulation::setCurrentTime(ct);
// 		if (gms.size())
// 		{
// 			foreach(object* o, gms)
// 			{
// 				if (ct >= o->MotionCondition().st)
// 					o->UpdateGeometryMotion(simulation::dt);
// 			}
// 		}
// 		if (dem)
// 		{
// 			model::isSinglePrecision ?
// 				dem->oneStepAnalysis_f(ct, cstep) :
// 				dem->oneStepAnalysis(ct, cstep);
// 			//qDebug() << "dem done";
// 		}
// 
// 		if (mbd)
// 		{
// 			mbd_state = mbd->oneStepAnalysis(ct, cstep);
// 			if (event_trigger::IsEvnetTrigger())
// 			{
// 				sendProgress(part, event_trigger::OnMessage());
// 			}
// 			if (mbd_state == -1)
// 			{
// 				//errors::Error(mbd->MbdModel()->modelName());
// 				//break;
// 			}
// 			else if (mbd_state == 1)
// 			{
// 				if (mg->ContactManager())
// 					mg->ContactManager()->update();// ContactParticlesPolygonObjects()->updatePolygonObjectData();
// 			}
// 		}
// 
// 		if (!((cstep) % simulation::st))
// 		{
// 			double dur_time = tme.elapsed() * 0.001;
// 			total_time += dur_time;
// 			part++;
// 			if (savePart(ct, part))
// 			{
// 				ch.clear();
// 				qts << qSetFieldWidth(0) << "| "
// 					<< qSetFieldWidth(nFit(15, part)) << part
// 					<< qSetFieldWidth(nFit(20, ct)) << ct
// 					<< qSetFieldWidth(nFit(20, eachStep)) << eachStep
// 					<< qSetFieldWidth(nFit(20, cstep)) << cstep
// 					<< qSetFieldWidth(nFit(20, dur_time)) << dur_time
// 					// 					<< qSetFieldWidth(17) << ct 
// 					// 					<< qSetFieldWidth(12) << eachStep 
// 					// 					<< qSetFieldWidth(13) << cstep 
// 					// 					/*<< qSetFieldWidth(22)*/ << dur_time 
// 					<< "|";
// 				sendProgress(part, ch);
// 				ch.clear();
// 			}
// 			eachStep = 0;
// 		}
// 	}
// 	model::rs->exportEachResult2TXT(model::path);
// 	saveFinalResult(ct);
// 	processResultData();
// 	sendProgress(0, "__line__");
// 	QTime endingTime = tme.currentTime();
// 	QDate endingDate = QDate::currentDate();
// 	double dtime = tme.elapsed() * 0.001;
// 	int minute = static_cast<int>(dtime / 60.0);
// 	int hour = static_cast<int>(minute / 60.0);
// 	qts.setFieldWidth(0);
// 	int cgtime = endingTime.second() - startingTime.msec();
// 	qts << "     Starting time/date     = " << startingTime.toString() << " / " << startingDate.toString() << endl
// 		<< "     Ending time/date      = " << endingTime.toString() << " / " << endingDate.toString() << endl
// 		<< "     CPU + GPU time       = " << dtime << " second  ( " << hour << " h. " << minute << " m. " << dtime << " s. )";
// 	sendProgress(-1, ch); ch.clear();
// 	emit finishedThread();
}