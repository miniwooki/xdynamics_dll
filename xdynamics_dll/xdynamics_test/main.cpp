//#include "xdynamics.h"
#include "xdynamics.h"
//#include "boost/algorithm.hpp"
//#include <QtCore/QString>
#include <crtdbg.h>

//static xMultiBodySimulation* xmbs = NULL;

int main(int argc, char* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//xModel::setModelName("fdafd");
//	QString
// 	std::cout << "Num. argc : " << argc << std::endl;
// 	std::wcout << argv[0] << ", " << argv[1] << std::endl;
// 	QMap<QString, xPointMass*> pms;
// 	pms["ddd"] = new xPointMass(L"ddd");
// 	pms["dsf"] = new xPointMass(L"dsf");
// 	xSparseD m(7,9);
// 	//m.alloc(7*9);
// 	m(1 - 1, 1 - 1) = m(2 - 1, 2 - 1) = m(3 - 1, 3 - 1) = m(4 - 1, 1 - 1) = m(5 - 1, 2 - 1) = m(7 - 1, 6 - 1) = 1;
// 	m(4 - 1, 4 - 1) = m(5 - 1, 5 - 1) = m(7 - 1, 9 - 1) = -1;
// 	m(4 - 1, 6 - 1) = 0.45;
// 	m(5 - 1, 6 - 1) = -0.21;
// 	m(6 - 1, 4 - 1) = -0.91;
// 	m(6 - 1, 5 - 1) = 0.42;
// 	m(6 - 1, 6 - 1) = -0.5;
// 	m(6 - 1, 7 - 1) = 0.91;
// 	m(6 - 1, 8 - 1) = -0.42;
// 	int* v = new int[2];
// 	coordinatePartitioning(m, v);
	xDynamicsManager *xdm = new xDynamicsManager;

//	if (argc != 1)	
//	{
//		/*xDynamicsSimulator* xds = *//*xdm.getSimulatorFromCommand(argc, argv);*/
//		if (xdm->getSimulatorFromCommand(argc, argv))
//		{
//// 			double dt = 0, et = 0;
//// 			unsigned int st = 0;Op
//// 			std::wcout
//// 				<< "Input the simulation conditions." << std::endl
//// 				<< "    - Time step : "; 
//// 			std::wcin >> dt;
//// 			std::wcout << "    - End time : ";
//// 			std::wcin >> et;
//// 			std::wcout << "    - Save step : ";
//// 			std::wcin >> st;
//// 			xds->xInitialize(dt, st, et);
//// 			xds->xRunSimulation();
//		}
//		else
//		{
//			std::string check_command = argv[1];
//			if (check_command == "-result")
//			{
//				xResultManager *xrm = new xResultManager;
//				if (argc > 2)
//				{
//					std::string check_model = argv[2];
//					xrm->xRun("C:/Users/xdynamics/Documents/xdynamics/", check_model);
//				}
//			/*	else
//				{
//					xrm->xRun("C:/Users/xdynamics/Documents/xdynamics", xUtilityFunctions::xstring(xModel::name));
//				}*/
//				//xdm.xRunResultWorld(argv[2]);
//			}
//			return 0;
//		}
//// 		delete xds;
//// 		return 0;
//	}
//	else
//	{
//// 		std::wstring check_command = argv[1];
//// 		if (check_command == L"-result")
//// 		{
////			xResultManager xrm;
////// 			if (argc > 2)
////// 			{
////				std::wstring check_model = L"one_pendulum";// argv[2];
////				xrm.xRun(xModel::path.toStdWString(), check_model);
//// 			}
//// 			else
//// 			{
//// 				xrm.xRun(xModel::path.toStdWString(), xModel::name.toStdWString());
//// 			}
//			//xdm.xRunResultWorld(argv[2]);
//		//}
//		//return 0;
//		xdm->OpenModelXLS("C:/xdynamics/resource/sphere_drop.xls");
//		//xdm->OpenModelXLS(L"C:/xDynamics/resource/four_bar3d.xls");
//	}
	
// 	xResultManager xrm;
// 	xrm.xRun(xModel::path, xModel::name);
 	//xIntegratorHHT* xmbs = new xIntegratorHHT;
	xdm->OpenModelXLS("C:/xdynamics/resource/cluster_mesh_contact_test/cluster_mesh_contact_test.xls");
 	xDynamicsSimulator *xds = new xDynamicsSimulator(xdm);
  	xds->xInitialize();
 	if (!xds->xRunSimulation())
 	{
 	//	xdynamicsReset();
 	}
	delete xds;
	delete xdm;
// 	//xmbs.initialize(xdm.XMBDModel());
// 	//ultiBodyModel mbd(L"model1");
// 	//xKinematicConstraint* rev = mbd.CreateRevoluteConstraint("rev");
// 	//rev->SetupParameters()
// 
// 	//	std::cout << m.Name() << std::endl;
// 	std::cout << c.x << " " << c.y << " " << c.z << std::endl;
	return 0;
}