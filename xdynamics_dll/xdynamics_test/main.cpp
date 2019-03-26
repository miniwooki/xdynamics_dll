#include "xdynamics.h"
//#include "boost/algorithm.hpp"
#include <crtdbg.h>

//static xMultiBodySimulation* xmbs = NULL;

int wmain(int argc, wchar_t* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//xModel::setModelName("fdafd");
//	QString
// 	std::cout << "Num. argc : " << argc << std::endl;
// 	std::wcout << argv[0] << ", " << argv[1] << std::endl;
// 	QMap<QString, xPointMass*> pms;
// 	pms["ddd"] = new xPointMass(L"ddd");
// 	pms["dsf"] = new xPointMass(L"dsf");
	xDynamicsManager *xdm = new xDynamicsManager;

	if (argc != 1)
	{
		/*xDynamicsSimulator* xds = *//*xdm.getSimulatorFromCommand(argc, argv);*/
		if (xdm->getSimulatorFromCommand(argc, argv))
		{
// 			double dt = 0, et = 0;
// 			unsigned int st = 0;
// 			std::wcout
// 				<< "Input the simulation conditions." << std::endl
// 				<< "    - Time step : "; 
// 			std::wcin >> dt;
// 			std::wcout << "    - End time : ";
// 			std::wcin >> et;
// 			std::wcout << "    - Save step : ";
// 			std::wcin >> st;
// 			xds->xInitialize(dt, st, et);
// 			xds->xRunSimulation();
		}
		else
		{
			std::wstring check_command = argv[1];
			if (check_command == L"-result")
			{
				xResultManager *xrm = new xResultManager;
				if (argc > 2)
				{
					std::wstring check_model = argv[2];
					xrm->xRun(xModel::path.toStdWString(), check_model);
				}
				else
				{
					xrm->xRun(xModel::path.toStdWString(), xModel::name.toStdWString());
				}
				//xdm.xRunResultWorld(argv[2]);
			}
			return 0;
		}
// 		delete xds;
// 		return 0;
	}
	else
	{
// 		std::wstring check_command = argv[1];
// 		if (check_command == L"-result")
// 		{
//			xResultManager xrm;
//// 			if (argc > 2)
//// 			{
//				std::wstring check_model = L"one_pendulum";// argv[2];
//				xrm.xRun(xModel::path.toStdWString(), check_model);
// 			}
// 			else
// 			{
// 				xrm.xRun(xModel::path.toStdWString(), xModel::name.toStdWString());
// 			}
			//xdm.xRunResultWorld(argv[2]);
		//}
		//return 0;
		xdm->OpenModelXLS(L"C:/xDynamics/resource/four_bar3d.xls");
	}
	
// 	xResultManager xrm;
// 	xrm.xRun(xModel::path, xModel::name);
 	//xIntegratorHHT* xmbs = new xIntegratorHHT;
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