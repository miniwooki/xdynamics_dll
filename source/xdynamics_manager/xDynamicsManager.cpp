#include "xdynamics_manager/xDynamicsManager.h"
#include "xdynamics_simulation/xDynamicsSimulator.h"
#include "xdynamics_simulation/xIntegratorHHT.h"
#include "xXLSReader.h"
#include "xViewExporter.h"
#include <QtCore/QDir>
#include <QtCore/QFile>
//#include <QtWidgets/QtWidgets>
#include <QtCore/QStringList>
#include <map>

xDynamicsManager::xDynamicsManager()
	: xModel("Model1")
	, xmbd(NULL)
	, xdem(NULL)
	, xsph(NULL)
	, xom(NULL)
	, xcm(NULL)
{
	CreateModel(xModel::name.toStdString(), OBJECT);
}

xDynamicsManager::~xDynamicsManager()
{
	qDeleteAll(xmbds);
	qDeleteAll(xdems);
	qDeleteAll(xsphs);
	qDeleteAll(xoms);
	qDeleteAll(xcms);
	xmbds.clear();
	xdems.clear();
	xsphs.clear();
	xoms.clear();
	xcms.clear();
	xmbd = NULL;
	xdem = NULL;
	xsph = NULL;
	xom = NULL;
	xcm = NULL;
	xSimulation::initialize();
	xObject::initialize();
}

bool xDynamicsManager::getSimulatorFromCommand(int argc, char* argv[])
{
	std::string argv1 = argv[1];
//	std::string check_command = xUtilityFunctions::WideChar2String(/*argv[1]*/);
	if (argv1 == "-result")
		return NULL;
	int nOpt = argc - 2;
	//QString file = xUtilityFunctions::WideChar2String(/*argv[1]*/);
	//size_t begin = argv1.find_last_of(".") + 1;
	QString ext = xUtilityFunctions::FileExtension(argv1.c_str());// .substr(begin, argv1.size());
	solverType stype;
	//xLog::log(ext.toStdString().c_str());
	if (ext == ".xls")
		stype = OpenModelXLS(argv1.c_str()/*argv[1]*/);
	else
	{
		xLog::log("Error : Unsupported file format.");
		return false;
	}
		
// 	xLog::log("Num. options " + QString("%1").arg(nOpt).toStdString());
// 	xDynamicsSimulator* xds = new xDynamicsSimulator(this);
// 	if (nOpt > 0)
// 	{
// 		bool default_set = false;
// 		for (int i = 2; i < argc; i++)
// 		{
// 			std::string opt_id = argv[i];// xUtilityFunctions::WideChar2String(argv[i]);
// 			std::string opt_value = argv[i + 1];// xUtilityFunctions::WideChar2String(argv[i + 1]);
// 			if (opt_id == "-default")
// 				default_set = true;
// 			if (opt_id == "-i")
// 			{
// 				if (opt_value == "hht")
// 				{
// 					xIntegratorHHT* hht = dynamic_cast<xIntegratorHHT*>(xds->setupMBDSimulation(xSimulation::IMPLICIT_HHT));
// 					char yorn;
// 					double alpha = 0.0, eps = 0.0;
// 					while (!default_set)
// 					{
// 						std::cout << "1. Default alpha : " << hht->AlphaValue() << std::endl;
// 						std::cout << "2. Default tolerance : " << hht->Tolerance() << std::endl;
// 						std::cout << "Do you want to edit it?(y/n) : ";
// 						std::wcin >> yorn;
// 						if (yorn == L'y')
// 						{
// 							unsigned int n = 0;
// 							std::cout << "Enter the number of the value to change : ";
// 							std::wcin >> n;
// 							if (n == 1)
// 							{
// 								std::cout << "Please enter a alpha value : ";
// 								std::wcin >> alpha;
// 								hht->setAlphaValue(alpha);
// 							}
// 							else if (n == 2)
// 							{
// 								std::cout << "Please enter a tolerance : ";
// 								std::wcin >> eps;
// 								hht->setImplicitTolerance(eps);
// 							}
// 							else
// 							{
// 								std::cout << "You have entered an invalid number." << std::endl;
// 							}
// 						}
// 						else
// 							break;
// 					}
// 					i++;
// 				}				
// 			}
// 		}
// 	}
// 	else
// 	{
// 		std::cout << "It will be simulation by setting as default." << std::endl;
// 		if (xmbd)
// 		{
// 			xds->setupMBDSimulation(xSimulation::IMPLICIT_HHT);
// 			std::cout << "    Default multibody solver : Implicit HHT\n" << std::endl;
// 		}			
// 	}
	return true;
}

xDiscreteElementMethodModel* xDynamicsManager::XDEMModel(std::string& _n)
{
	QString n = QString::fromStdString(_n);
	QStringList keys = xdems.keys();
	QStringList::const_iterator it = qFind(keys, n);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xdems[n];
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
	QString nm = QString::fromStdString(n);
	QStringList keys = xsphs.keys();
	QStringList::const_iterator it = qFind(keys, nm);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xsphs[nm];
}

xObjectManager* xDynamicsManager::XObject()
{
	return xom;
}

xObjectManager* xDynamicsManager::XObject(std::string& _n)
{
	QString n = QString::fromStdString(_n);
	QStringList keys = xoms.keys();
	QStringList::const_iterator it = qFind(keys, n);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xoms[n];
}

xContactManager* xDynamicsManager::XContact()
{
	return xcm;
}

xContactManager* xDynamicsManager::XContact(std::string& _n)
{
	QString n = QString::fromStdString(_n);
	QStringList keys = xcms.keys();
	QStringList::const_iterator it = qFind(keys, n);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xcms[n];
}

xMultiBodyModel* xDynamicsManager::XMBDModel()
{
	return xmbd;
}

xMultiBodyModel* xDynamicsManager::XMBDModel(std::string& _n)
{
	QString n = QString::fromStdString(_n);
	QStringList keys = xmbds.keys();
	QStringList::const_iterator it = qFind(keys, n);
	if (it == keys.end() || !keys.size())
		return NULL;
	return xmbds[n];
}

xDynamicsManager::solverType xDynamicsManager::OpenModelXLS(const char* n)
{
	xXLSReader xls;
	if (xls.Load(n))
	{
		QString file_name = xUtilityFunctions::GetFileName(n);
		xModel::setModelName(file_name);
		QString md = xls.SetupSheet(0);
		std::map<xXlsInputDataType, vector2i> xx;
		int c = 2;
		while (!xls.IsEmptyCell(0, c))
		{
			vector2i d;
			QString tn = xls.ReadStr(0, c++);
			std::string t = xls.ReadStr(0, c++).toStdString();
			xUtilityFunctions::xsplit(t, ",", d);
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
 		std::string model_name = xModel::name.toStdString();
 		std::string full_path = xModel::path.toStdString() + model_name + "/" + model_name;
 		//xUtilityFunctions::DeleteFilesInDirectory(xUtilityFunctions::xstring(xModel::path) + model_name);
 		//xUtilityFunctions
		QString dDir = QString::fromStdString(full_path);
		QDir dir = QDir(dDir);
		QStringList delFileList;
		delFileList = dir.entryList(QStringList("*.*"), QDir::Files | QDir::NoSymLinks);
		//qDebug() << "The number of *.bin file : " << delFileList.length();
		for (int i = 0; i < delFileList.length(); i++){
			QString deleteFilePath = dDir + delFileList[i];
			QFile::remove(deleteFilePath);
		}
		xViewExporter xve;

		xve.Open(full_path + ".vmd");
		xls.setViewExporter(&xve);
		//xls.CreateViewModelOutput(full_path + ".vmd");

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
						//xsph->CreateParticles(xom);
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
// 		if (xsph)
// 		{
// 			//xom->CreateSPHBoundaryParticles(xsph->XParticleManager());
// 			xsph->
// 		}
		if (xdem || xsph)
		{
			std::string pv_path = full_path + ".par";
			if(xdem) xdem->XParticleManager()->ExportParticleDataForView(pv_path);
			//else if (xsph) xsph->ExportParticleDataForView(pv_path);
			int vot = VPARTICLE;
			xve.Write((char*)&vot, sizeof(xViewObjectType));
			int ns = pv_path.size(); xve.Write((char*)&ns, sizeof(int));
			xve.Write((char*)pv_path.c_str(), sizeof(char)*pv_path.size());
		}
// 		foreach(xObject* xo, xom->XObjects())
// 		{
// 			if (xo->Shape() == MESH_SHAPE)
// 			{
// 				xMeshObject* xmo = dynamic_cast<xMeshObject*>(xo);
// 				QString mname = xmo->Name() + ".mesh";
// 				std::string file = xmo->exportMeshData(xModel::makeFilePath(mname.toStdString()));
// 			}
// 		}
 		xve.Close();
	}
	solverType stype= ONLY_MBD;
	if (xmbd && xdem) stype = COUPLED_MBD_DEM;
	else if (xmbd) stype = ONLY_MBD;
	else if (xdem) stype = ONLY_DEM;
	return stype;
}

void xDynamicsManager::CreateModel(std::string n, modelType t, bool isOnAir /*= false*/)
{
	QString qname = xModel::name;
	switch (t)
	{
	case MBD: xmbds[qname] = new xMultiBodyModel(n); break;
	case DEM: xdems[qname] = new xDiscreteElementMethodModel(n); break;
	case SPH: xsphs[qname] = new xSmoothedParticleHydrodynamicsModel(n); break;
	case OBJECT: xoms[qname] = new xObjectManager(); break;
	case CONTACT: xcms[qname] = new xContactManager(); break;
	}

	if (isOnAir)
		setOnAirModel(t, qname.toStdString());
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

