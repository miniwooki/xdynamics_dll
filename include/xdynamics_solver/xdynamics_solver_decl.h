#ifndef XDYNAMICS_DECL_H
#define XDYNAMICS_DECL_H

#ifdef XDYNAMICS_STDCALL
#define XDYNAMICS_CALLING __stdcall
#else
#define XDYNAMICS_CALLING __cdecl
#endif

#ifndef XDYNAMICS_EXPORTS
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __declspec(dllimport)
#endif

#define XDAPIENTRY XDYNAMICS_CALLING

#include <iostream>
#include <windows.h>

using namespace std;

#endif