#include "xCommandLine.h"
#include "xdynamics_algebra/xUtilityFunctions.h"
//#include "modelManager.h"
#include <QTextStream>

// int xCommandLine::step0(int c, QString s)
// {
// 	if (c == 1) // event
// 	{
// 		if (s == "mbd")
// 			if (sz > 2)
// 				return step1(11, sList.at(++cidx));
// 			else
// 				return 11;
// 		if (s == "contact")
// 			if (sz > 2)
// 				return step1(12, sList.at(++cidx));
// 			else
// 				return 12;
// 	}
// 	else if (c == 2) // RV
// 	{
// 		mbd_model* mbd = modelManager::MM()->MBDModel();
// 		if (mbd)
// 		{
// 			mass = mbd->PointMass(s);
// 			if (mass)
// 				if (sz == 5)
// 				{
// 					VEC3D rv(sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble());
// 					mass->setRotationalVelocity(rv);
// 					//return step1(21, sList.at(++cidx));
// 				}
// 				else
// 					return 21;
// 		}
// 		return 20;
// 	}
// 	else
// 		return c;
// 	return -1;
// }
// 
// int xCommandLine::step1(int c, QString s)
// {
// 	if (c == 11) // event -> mbd
// 	{
// 		if (s == "start_time")
// 			if (sz > 3)
// 				return step2(111, sList.at(++cidx));
// 			else
// 				return 111;
// 	}
// 	else if (c == 12) // event -> contact
// 	{
// 		contactManager* cm = modelManager::MM()->ContactManager();
// 		if (cm)
// 		{
// 			cont = cm->Contact(s);
// 			if (cont)
// 				if (sz > 3)
// 					return step2(121, sList.at(++cidx));
// 				else
// 					return 121;
// 		}
// 		return 120;
// 	}
// 	// 	else if (c == 21)
// 	// 	{
// 	// 		if (sList.size() != 5)
// 	// 			return -1;
// 	// 		VEC3D rv(s.toDouble(), sList.at(++cidx).toDouble(), sList.at(++cidx).toDouble());
// 	// 		mass = 
// 	// 	}
// 	return -1;
// }
// 
// int xCommandLine::step2(int c, QString s)
// {
// 	if (c == 111) // event -> mbd -> start_time
// 	{
// 		mbd_model* mbd = modelManager::MM()->MBDModel();
// 		if (mbd)
// 			mbd->setStartTimeForSimulation(s.toDouble());
// 		return 0;
// 	}
// 	else if (c == 121)
// 	{
// 		if (s == "ignore_time")
// 			if (sz > 4)
// 				return step3(1211, sList.at(++cidx));
// 			else
// 				return 1211;
// 	}
// 	return -1;
// }
// 
// int xCommandLine::step3(int c, QString s)
// {
// 	if (c == 1211)
// 	{
// 		if (cont)
// 			cont->setIgnoreTime(s.toDouble());
// 		cont = NULL;
// 		return 0;
// 	}
// 
// 	return -1;
// }

xCommandLine::xCommandLine()
	: is_finished(false)
	, isOnCommand(false)
	, isWrongCommand(false)
	, sz(0)
	, cidx(0)
	, cstep(0)
	, current_log_index(0)
	, cylinder(NULL)
	, cube(NULL)
{

}

xCommandLine::~xCommandLine()
{
	if (cylinder) delete cylinder; cylinder = NULL;
	if (cube) delete cube; cube = NULL;
}

bool xCommandLine::IsFinished()
{
	return is_finished;
}

bool xCommandLine::IsWrongCommand()
{
	return isWrongCommand;
}

void xCommandLine::SetCurrentAction(int i)
{
	if (isOnCommand) return;
	switch (i)
	{
	case 3: cube = new xCubeObjectData; break;
	case 4: cylinder = new xCylinderObjectData; break; // 3 is cylinder action
	default:
		break;
	}
	cstep = 0;
	isOnCommand = true;
}

QString xCommandLine::getPassedCommand()
{
// 	QString c;
// 	int sz_log = logs.size();
// 	if (!current_log_index)
// 		current_log_index = sz_log;
// 	if (sz_log)
// 	{
// 		c = logs.at(current_log_index - 1);
// 	}
 	return "";
}

QString xCommandLine::CylinderCommandProcess(QString& com)
{
	isWrongCommand = false;
	if (cylinder)
	{
		if (cstep == 0)
		{
			QStringList s = com.split(',');
			if (s.size() != 3)
			{
				isWrongCommand = true;
				return "Invalid command";
			}
			cylinder->p0x = s.at(0).toDouble();
			cylinder->p0y = s.at(1).toDouble();
			cylinder->p0z = s.at(2).toDouble();
			cstep++;
			return "Input the bottom point.";
		}
		else if (cstep == 1)
		{
			QStringList s = com.split(',');
			if (s.size() != 3)
			{
				isWrongCommand = true;
				return "Invalid command";
			}
			cylinder->p1x = s.at(0).toDouble();
			cylinder->p1y = s.at(1).toDouble();
			cylinder->p1z = s.at(2).toDouble();
			cstep++;
			return "Input the top radius.";
		}
		else if (cstep == 2)
		{
			cylinder->r_top = com.toDouble();
			cstep++;
			return "Input the bottom radius.";
		}
		else if (cstep == 3)
		{
			cylinder->r_bottom = com.toDouble();
			cstep++;
			is_finished = true;
			return "Command Line";
		}
	}
	isWrongCommand = true;
	return "Invalid command";
}

QString xCommandLine::CubeCommandProcess(QString& com)
{
	isWrongCommand = false;
	if (cube)
	{
		if (cstep == 0)
		{
			QStringList s = com.split(',');
			if (s.size() != 3)
			{
				isWrongCommand = true;
				return "Invalid command";
			}
			cube->p0x = s.at(0).toDouble();
			cube->p0y = s.at(1).toDouble();
			cube->p0z = s.at(2).toDouble();
			cstep++;
			return "Input the maximum point.";
		}
		else if (cstep == 1)
		{
			QStringList s = com.split(',');
			if (s.size() != 3)
			{
				isWrongCommand = true;
				return "Invalid command";
			}
			cube->p1x = s.at(0).toDouble();
			cube->p1y = s.at(1).toDouble();
			cube->p1z = s.at(2).toDouble();
			cstep++;
			is_finished = true;
			return "Command Line";
		}
	}
	isWrongCommand = true;
	return "Invalid command";
}

xCylinderObjectData xCommandLine::GetCylinderParameters()
{
	xCylinderObjectData d;
	if (is_finished)
	{
		d = *cylinder;
		isOnCommand = false;
		delete cylinder; cylinder = NULL;
		
	}
	return d;
}

xCubeObjectData xCommandLine::GetCubeParameters()
{
	xCubeObjectData d;
	if (is_finished)
	{
		d = *cube;
		isOnCommand = false;
		delete cube; cube = NULL;

	}
	return d;
}
