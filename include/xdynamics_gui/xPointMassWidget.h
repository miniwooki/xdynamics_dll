#ifndef WPOINTMASS_H
#define WPOINTMASS_H

#include "ui_wpointmass.h"

class xPointMass;

class wpointmass : public QWidget, public Ui::wpointmass
{
	Q_OBJECT
public:
	wpointmass(QWidget* parent = NULL);
	~wpointmass();

	bool& IsOnConnectGeomegry() { return isOnConnectGeomegry; }

	public slots:
	void UpdateInformation(xPointMass* xpm);

signals:
	void clickEnableConnectGeometry(bool);

	private slots:
	void EnableConnectGeometry();

private:
	bool isOnConnectGeomegry;
};

#endif