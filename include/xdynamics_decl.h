#ifndef XDYNAMICS_DECL_H
#define XDYNAMICS_DECL_H

#ifdef XDYNAMICS_STDCALL
#define XDYNAMICS_CALLING __stdcall
#else
#define XDYNAMICS_CALLING __cdecl
#endif

#ifdef XDYNAMICS_DLL_EXPORTS
#define XDYNAMICS_API __declspec(dllexport)
#else
#define XDYNAMICS_API __declspec(dllimport)
#endif

#define XDYNAMICS_APIENTRY XDYNAMICS_CALLING

#define VERSION_NUMBER 1

#include <iostream>
#include <windows.h>
#include <cassert>

#include "xTypes.h"
#include "xdynamics_algebra/xAlgebraType.h"
//#include "xstring.h"
//#include "xmap.hpp"
//#include "xlist.hpp"

using namespace std;

/*#define kor(str) QString::fromLocal8Bit(str)*/

#endif