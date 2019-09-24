#include "xdynamics_manager/xDynamicsManager.h"
#include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_simulation/xIntegratorHHT.h"
#include "xXLSReader.h"
#include "xViewExporter.h"
//#include <QtCore/QDir>
//#include <QtCore/QFile>
////#include <QtWidgets/QtWidgets>
//#include <QtCore/QStringList>
#include <map>
#include <typeinfo>
#include <stdexcept>
#include <sstream>

static xDynamicsManager* xdmanager;

xDynamicsManager::xDynamicsManager()
	: xModel("Model1")
	, xmbd(NULL)
	, xdem(NULL)
	, xsph(NULL)
	, xom(NULL)
	, xcm(NULL)
	, xrm(NULL)
{
	CreateModel(xModel::name.toStdString(), OBJECT);
	xdmanager = this;
}

xDynamicsManager::~xDynamicsManager()
{
	if (xmbds.size()) xmbds.delete_all();// qDeleteAll(xmbds);
	if (xdems.size()) xdems.delete_all();//  qDeleteAll(xdems);
	if (xsphs.size()) xsphs.delete_all();
	if(xoms.size()) xoms.delete_all();
;	if (xcms.size()) xcms.delete_all();// qDeleteAll(xcms);
	xmbd = NULL;
	xdem = NULL;
	xsph = NULL;
	xom = NULL;
	xcm = NULL;
	if (xrm) delete xrm; xrm = NULL;
	xSimulation::initialize();
	xObject::initialize();
}

xDynamicsManager* xDynamicsManager::This()
{
	return xdmanager;
}

bool xDynamicsManager::getSimulatorFromCommand(int argc, char* argv[])
{
	std::string argv1 = argv[1];
//	std::string check_command = xUtilityFunctions::WideChar2String(/*argv[1]*/);
	if (argv1 == "-result")
		return NULL;
	int nOpt = argc - 2;
	std::string ext = xUtilityFunctions::FileExtension(argv1.c_str());// .substr(begin, argv1.size());
	//solverType stype;
	if (ext == ".xls")
		OpenModelXLS(argv1.c_str()/*argv[1]*/);
	else
	{
		xLog::log("Error : Unsupported file format.");
		return false;
	}
	return true;
}

xDiscreteElementMethodModel* xDynamicsManager::XDEMModel(std::string& _n)
{
	//QString n = QString::fromStdString(_n);
	//QStringList keys = xdems.keys();
	if (xdems.size() == 0)
		return NULL;
	xmap<xstring, xDiscreteElementMethodModel*>::iterator it = xdems.find(_n);//QStringList::const_iterator it = qFind(keys, n);
	if (it == xdems.end())
		return NULL;
	return it.value();// xdems[n];
}

xDiscreteElementMethodModel* xDynamicsManager::XDEMModel()
{
	return xdem;
}

xSmoothedParticleHydrodynamicsModel* xDynamicsManager::XSPHModel()
{
	return xsph;
}

xSmoothedParticleHydrodynamicsModel* xDynamicsManager::XSPHModel(std::string& n)
{
	//QString nm = QString::fromStdString(n);
	//QStringList keys = xsphs.keys();
	if (xsphs.size() == 0)
		return NULL;
	xmap<xstring, xSmoothedParticleHydrodynamicsModel*>::iterator it = xsphs.find(n);//QStringList::const_iterator it = qFind(keys, nm);
	if (it == xsphs.end())
		return NULL;
	return it.value();
}

xObjectManager* xDynamicsManager::XObject()
{
	return xom;
}

xObjectManager* xDynamicsManager::XObject(std::string& _n)
{
	if(xoms.find(_n) == xoms.end())
		return NULL;
	return xoms[_n];
}

xContactManager* xDynamicsManager::XContact()
{
	return xcm;
}

xResultManager * xDynamicsManager::XResult()
{
	return xrm;
}

xContactManager* xDynamicsManager::XContact(std::string& _n)
{
	//QString n = QString::fromStdString(_n);
	//QStringList keys = xcms.keys();
	//QStringList::const_iterator it = qFind(keys, n);
	if (xcms.size() == 0)
		return NULL;
	xmap<xstring, xContactManager*>::iterator it = xcms.find(_n);
	if (it == xcms.end())
		return NULL;
	return it.value();// xcms[n];
}

xMultiBodyModel* xDynamicsManager::XMBDModel()
{
	return xmbd;
}

xMultiBodyModel* xDynamicsManager::XMBDModel(std::string& _n)
{
	//QString n = QString::fromStdString(_n);
	//QStringList keys = xmbds.keys();
	//QStringList::const_iterator it = qFind(keys, n);
	if (xmbds.size() == 0)
		return NULL;
	xmap<xstring, xMultiBodyModel*>::iterator it = xmbds.find(_n);
	if (it == xmbds.end())
		return NULL;
	return it.value(); //xmbds[n];
}

int xDynamicsManager::OpenModelXLS(const char* n)
{
	xXLSReader xls;
	if (xls.Load(n))
	{
		std::string file_name = xUtilityFunctions::GetFileName(n);
		xModel::setModelName(file_name);
		xstring md = xls.SetupSheet(0);
		std::map<xXlsInputDataType, vector2i> xx;
		int c = 2;
		while (!xls.IsEmptyCell(0, c))
		{
			vector2i d;
			xstring tn = xls.ReadStr(0, c++);
			xstring t = xls.ReadStr(0, c++);
			t.split(",", 2, &d.x);
			d.x -= 1; d.y -= 1;
			if (tn == "SHAPE") xx[XLS_SHAPE] = d;
			else if (tn == "MASS") xx[XLS_MASS] = d;
			else if (tn == "JOINT") xx[XLS_JOINT] = d;
			else if (tn == "FORCE") xx[XLS_FORCE] = d;
			else if (tn == "PARTICLE") xx[XLS_PARTICLE] = d;
			else if (tn == "CONTACT") xx[XLS_CONTACT] = d;
			else if (tn == "KERNE") xx[XLS_KERNEL] = d;
			else if (tn == "INTEGRATOR") xx[XLS_INTEGRATOR] = d;
			else if (tn == "SIMULATION") xx[XLS_SIMULATION] = d;
			else if (tn == "GRAVITY") xx[XLS_GRAVITY] = d;
		}
		std::map<xXlsInputDataType, vector2i>::iterator bt = xx.begin();
		std::map<xXlsInputDataType, vector2i>::iterator et = xx.end();
		for (; bt != et; bt++)
		{
			bool is_empty_cell = xls.IsEmptyCell(bt->second.x, bt->second.y);
			if (is_empty_cell)
			{
				std::string s;
				stringstream ss(s);
				ss << "Exception in excel reader : Information cell[" << bt->second.x << ", " << bt->second.y << "] of " << NameOfXLSPart(bt->first) << " is empty.";
				xLog::log(ss.str().c_str());
				return xDynamicsError::xdynamicsErrorExcelModelingData;
			}
		}
		bt = xx.begin();
 		std::string model_name = xModel::name.toStdString();
 		std::string full_path = xModel::path.toStdString() + model_name + "/" + model_name;
		std::string dDir = full_path;
		xViewExporter xve;

		xve.Open(full_path + ".vmd");
		
		xls.setViewExporter(&xve);
		try
		{
			for (; bt != et; bt++)
			{
				switch (bt->first)
				{
				case XLS_SHAPE: xls.ReadShapeObject(xom, bt->second); break;
				case XLS_MASS:
					if (!this->XMBDModel(model_name))
					{
						CreateModel(model_name, MBD);
						xls.ReadMass(xmbd, bt->second);
					}break;
				case XLS_JOINT: xls.ReadJoint(xmbd, bt->second); break;
				case XLS_FORCE: xls.ReadForce(xmbd, xdem, bt->second); break;
				case XLS_KERNEL:
					if (!this->XSPHModel(model_name))
					{
						CreateModel(model_name, SPH);
						xls.ReadKernel(xsph, bt->second); break;
					}
				case XLS_PARTICLE:
					if (!this->XDEMModel(model_name))
					{
						if (xsph)
						{
							xls.ReadSPHParticle(xsph, bt->second);
							xsph->CreateParticles(xom);
						}
						else
						{
							CreateModel(model_name, DEM);
							xls.ReadDEMParticle(xdem, xom, bt->second);
						}
					}break;
				case XLS_CONTACT:
					if (!this->XContact(model_name))
					{
						CreateModel(model_name, CONTACT);
						xls.ReadContact(xcm, bt->second);
					} break;
				case XLS_INTEGRATOR: xls.ReadIntegrator(bt->second); break;
				case XLS_SIMULATION: xls.ReadSimulationCondition(bt->second); break;
				case XLS_GRAVITY: xls.ReadInputGravity(bt->second); break;
				}
			}
		}
		catch (exception &e)
		{
			xLog::log("Exception in excel reader : " + std::string(e.what()));
			xve.Close();
			return xDynamicsError::xdynamicsErrorExcelModelingData;
		}
		if (xdem || xsph)
		{
			std::string pv_path = full_path + ".par";
			if(xdem) xdem->XParticleManager()->ExportParticleDataForView(pv_path);
			//else if (xsph) xsph->ExportParticleDataForView(pv_path);
			int vot = VPARTICLE;
			xve.Write((char*)&vot, sizeof(xViewObjectType));
			int ns = pv_path.size(); xve.Write((char*)&ns, sizeof(int));
			xve.Write((char*)pv_path.c_str(), sizeof(char)*pv_path.size());
			if (xSimulation::dt == 0.0)
			{
				xParticleManager *xpm = xdem->XParticleManager();
				double new_dt = xUtilityFunctions::CriticalTimeStep(
					xpm->CriticalRadius(),
					xpm->CriticalDensity(),
					xpm->CriticalYoungs(),
					xpm->CriticalPoisson());
				xSimulation::setTimeStep(new_dt);
			}
		}
 		xve.Close();
	}
	return xDynamicsError::xdynamicsSuccess;
}

bool xDynamicsManager::upload_model_results(std::string path)
{
	if (xrm)
		delete xrm;
	xrm = new xResultManager;
	xrm->initialize_from_exist_results(path);
	if (xmbd)
	{
		for (xmap<xstring, xPointMass*>::iterator it = xmbd->Masses().begin(); it != xmbd->Masses().end(); it.next())
		{
			xstring name = it.key();
			xrm->alloc_mass_result_memory(name.toStdString());
		}
		for (xmap<xstring, xKinematicConstraint*>::iterator it = xmbd->Joints().begin(); it != xmbd->Joints().end(); it.next())
		{
			xstring name = it.key();
			xrm->alloc_joint_result_memory(name.toStdString());
		}
		for (xmap<xstring, xDrivingConstraint*>::iterator it = xmbd->Drivings().begin(); it != xmbd->Drivings().end(); it.next())
		{
			xstring name = it.key();
			xrm->alloc_joint_result_memory(name.toStdString());
		}
	}
//	xrm->upload_exist_results(path);
	return true;
}

void xDynamicsManager::CreateModel(std::string n, modelType t, bool isOnAir /*= false*/)
{
	//xstring qname = xModel::name;
	switch (t)
	{
	case MBD: xmbds.insert(n, new xMultiBodyModel(n)); break;
	case DEM: xdems.insert(n, new xDiscreteElementMethodModel(n)); break;
	case SPH: xsphs.insert(n, new xSmoothedParticleHydrodynamicsModel(n)); break;
	case OBJECT: xoms.insert(n, new xObjectManager()); break;//xoms[n] = new xObjectManager(); break;
	case CONTACT: xcms.insert(n, new xContactManager()); break;
	}

	if (isOnAir)
		setOnAirModel(t, n);
}

void xDynamicsManager::initialize_result_manager(unsigned int npt)
{
	if (xrm)
		delete xrm;
	xrm = new xResultManager();
	xrm->set_num_parts(npt);
	xrm->alloc_time_momory(npt);
}

//void xDynamicsManager::allocation_simulation_result_memory()
//{
//}

void xDynamicsManager::release_result_manager()
{
	if (xrm)
	{
		delete xrm;
		xrm = NULL;
	}
}

void xDynamicsManager::setOnAirModel(modelType t, std::string n)
{
	switch (t)
	{
	case DEM: xdem = XDEMModel(n); break;
	case MBD: xmbd = XMBDModel(n); break;
	case SPH: xsph = XSPHModel(n); break;
	case OBJECT: xom = XObject(n); break;
	case CONTACT: xcm = XContact(n); break;
	}
}

