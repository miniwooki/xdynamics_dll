#include "xdynamics_manager/xResultManager.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
#include "xdynamics_object/xPointMass.h"
#include "xdynamics_object/xKinematicConstraint.h"

xResultManager::xResultManager()
{

}

xResultManager::~xResultManager()
{

}

void xResultManager::xRun(const std::string _cpath, const std::string _cname)
{
	char cmd[64] = { 0, };
	
	cur_path = QString::fromStdString(_cpath);
	cur_name = QString::fromStdString(_cname);
	std::string _path = _cpath + _cname + "/";
	std::cout << "Welcome to result world." << std::endl;
	std::cout << "Current path - " << _path.c_str() << std::endl;

	xUtilityFunctions::DirectoryFileList(_path.c_str());
	int ret = 0;
	int ncmd = 0;
	while (1)
	{
		ret = 0;
		std::cout << ">> ";
		std::cin.getline(cmd, sizeof(cmd), '\n');
		ncmd = xUtilityFunctions::xsplitn(cmd, " ");
		switch (ncmd)
		{
		case 0: ret = Execute0(cmd); break;
		case 1: ret = Execute1(cmd); break;
		case 2: ret = Execute2(cmd); break;
		}
	
		if (!ret)
			std::cout << "The command you entered does not exist." << std::endl;
		else if (ret == -1)
			break;
		fflush(stdin);
	}
}

void xResultManager::setCurrentPath(std::string new_path)
{
	cur_path = QString::fromStdString(new_path);// wsprintfW(cur_path, TEXT("%s"), new_path);
}

void xResultManager::setCurrentName(std::string new_name)
{
	cur_name = QString::fromStdString(new_name);// wsprintfW(cur_name, TEXT("%s"), new_name);
}

void xResultManager::ExportBPM2TXT(std::string& file_name)
{
	std::fstream ifs;
	ifs.open(file_name.c_str(), ios::binary | ios::in);
	//xLog::log(xUtilityFunctions::WideChar2String(file_name.c_str()));
	int identifier;
	unsigned int nr_part;
	char t;
	ifs.read((char*)&identifier, sizeof(int));
	ifs.read((char*)&t, sizeof(char));
	ifs.read((char*)&nr_part, sizeof(unsigned int));
	xPointMass::pointmass_result *pms = new xPointMass::pointmass_result[nr_part];
	ifs.read((char*)pms, sizeof(xPointMass::pointmass_result) * nr_part);
	ifs.close();
	//size_t begin = file_name.find_last_of(".");
	size_t end = file_name.find_last_of("/");
	std::string new_file_name = file_name.substr(0, end + 1) + xUtilityFunctions::GetFileName(file_name.c_str()).toStdString() + ".txt";
	ifs.open(new_file_name, ios::out);
	ifs << "time " << "px " << "py " << "pz " << "vx " << "vy " << "vz " << "ax " << "ay " << "az "
		<< "avx " << "avy " << "avz " << "aax " << "aay " << "aaz "
		<< "afx " << "afy " << "afz " << "amx " << "amy " << "amz "
		<< "cfx " << "cfy " << "cfz " << "cmx " << "cmy " << "cmz "
		<< "hfx " << "hfy " << "hfz " << "hmx " << "hmy " << "hmz "
		<< "ep0 " << "ep1 " << "ep2 " << "ep3 "
		<< "ev0 " << "ev1 " << "ev2 " << "ev3 "
		<< "ea0 " << "ea1 " << "ea2 " << "ea3" << std::endl;

	for (unsigned int i = 0; i < nr_part; i++)
	{
		for (unsigned int j = 0; j < 46; j++)
		{
			double v = *(&(pms[0].time) + i * 46 + j);
			ifs << v << " ";
		}
		ifs << std::endl;
	}
	ifs.close();
}

void xResultManager::ExportBKC2TXT(std::string& file_name)
{
	std::fstream ifs;
	ifs.open(file_name, ios::binary | ios::in);
	int identifier;
	unsigned int nr_part;
	char t;
	ifs.read((char*)&identifier, sizeof(int));
	ifs.read((char*)&t, sizeof(char));
	ifs.read((char*)&nr_part, sizeof(unsigned int));
	xKinematicConstraint::kinematicConstraint_result *pms = new xKinematicConstraint::kinematicConstraint_result[nr_part];
	ifs.read((char*)pms, sizeof(xKinematicConstraint::kinematicConstraint_result) * nr_part);
	ifs.close();
	//size_t begin = file_name.find_last_of(".");
	std::string new_file_name = xUtilityFunctions::GetFileName(file_name.c_str()).toStdString() + ".txt";//.substr(0, begin) + ".txt";
	ifs.open(new_file_name, ios::out);
	ifs << "time " << "locx " << "locy " << "locz "// << "vx " << "vy " << "vz " << "ax " << "ay " << "az "
		<< "iafx " << "iafy " << "iafz " << "irfx " << "irfy " << "irfz "
		<< "jafx " << "jafy " << "jafz " << "jrfx " << "jrfy " << "jrfz ";

	for (unsigned int i = 0; i < nr_part; i++)
	{
		for (unsigned int j = 0; j < 16; j++)
		{
			double v = *(&(pms[0].time) + i * 16 + j);
			ifs << v << " ";
		}
		ifs << std::endl;
	}
	ifs.close();
}

int xResultManager::Execute0(char *d)
{
	return 1;
}

int xResultManager::Execute1(char *d)
{
	if (!strcmp("exit", d))
		return -1;
	else if (!strcmp("list", d))
	{
		xUtilityFunctions::DirectoryFileList((cur_path + cur_name).toStdString().c_str());
		return 1;
	}
	return 0;
}

int xResultManager::Execute2(char *d)
{
	char val[64] = { 0, };
	std::string data[2];
	xUtilityFunctions::xsplit(d, " ", 2, data);
	if (data[0] == "get")
	{
		if (data[1] == "ascii")
		{
			std::cout << "Please enter a result file to import : ";
			std::cin >> val;
			std::string fn = (cur_path + cur_name + "/").toStdString() + val;
			if (xUtilityFunctions::ExistFile(fn.c_str()))
			{
				QString ext = xUtilityFunctions::FileExtension(fn.c_str());
				if (ext == ".bpm")
					ExportBPM2TXT(fn);
				else if (ext == ".bkc")
					ExportBKC2TXT(fn);
			}
			return 1;
		}
	}
	else if (data[0] == "set")
	{
		if (data[1] == "mode")
		{
			std::cout << "Please enter a model name : ";
			std::cin >> val;
			std::string cname = val;
			std::string fn = cur_path.toStdString() + val + "/";
			if (xUtilityFunctions::ExistFile(fn.c_str()))
			{
				cur_name = QString::fromStdString(cname);
			//	_path = fn;
				xUtilityFunctions::DirectoryFileList(fn.c_str());
			}
			else
			{
				std::cout << "The model you entered does not exist." << std::endl;
			}
			return 1;
		}
	}
	return 0;
}

