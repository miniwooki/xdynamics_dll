#pragma once

#include "xdynamics_object/xObject.h"
#include "xlist.hpp"

class xMassSpringModel : public xObject
{
public:
	xMassSpringModel();
	xMassSpringModel(std::string _name);
	~xMassSpringModel();

private:
	//unsigned int nsdci;
	//unsigned int nkcvalue;
	//unsigned int nConnection;
	//unsigned int nBodyConnection;
	//unsigned int nBodyConnectionData;
	xlist<xSpringDamperCoefficient> kc_value;
	xlist<xSpringDamperConnectionInformation> xsdci;
	xlist<xSpringDamperConnectionData> connection_data;
	xlist<xSpringDamperBodyConnectionInfo> connection_body_info;
	xlist<xSpringDamperBodyConnectionData> connection_body_data;
	//double *free_length;
};
