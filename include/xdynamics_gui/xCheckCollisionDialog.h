#pragma once

#include "ui_check_collision_dialog.h"
#include "xTypes.h"
#include <QColor>
#include <QMap>

class xNeiborhoodCell;
class xContactManager;
class xParticleManager;
class xObjectManager;
class xvParticle;

class xCheckCollisionDialog : public QDialog, public Ui::CheckCollisionDialog
{
	Q_OBJECT
	typedef struct
	{
		double overlap;
		double ux, uy, uz;
	}xCollisionPair;
public:
	xCheckCollisionDialog(QWidget* parent = nullptr);
	~xCheckCollisionDialog();

	void setup(xvParticle* _xp, xParticleManager* _xpmgr, xObjectManager* _omgr);

protected:
	bool checkTreeItemHasChild(QTreeWidgetItem* item);
	void updateCollision();
	void mouseReleaseEvent(QMouseEvent *event);
	void MoveParticle(unsigned int id, double x, double y, double z);
	

private slots:
	void checkCollision();
	void clickCancel();
	void clickTreeItem(QTreeWidgetItem* item, int column);
	void highlightSelectedCluster(int row, int column);
	void selectedItemProcess();
	void movePlusX();
	void movePlusY();
	void movePlusZ();
	void moveMinusX();
	void moveMinusY();
	void moveMinusZ();
	void movePlusNormal();
	void moveMinusNormal();
	//void changePosition(QTableWidgetItem* item);

//signals:
//	void highlightSelectedParticle(unsigned int id, QColor rgb, QColor&);

private:
	bool isSetup;
	bool isChangedSelection;
	int selectedCluster;
	xvParticle* xp;
	xObjectManager* omgr;
	xParticleManager* pmgr;
	QList<QListWidgetItem*> items;
	//QPair<unsigned int, unsigned int> selectedPair;
	QMap<QString, xClusterInformation> cinfos;
	QMap<QString, QTreeWidgetItem*> iclusters;
	QMap<QString, QTreeWidgetItem*> jclusters;
	QMap<QString, xCollisionPair> collision_info;
	QMap<unsigned int, QColor> changedColor;
	//QMap<unsigned int, QTableWidgetItem*> tableItems;
	//QMap<QPair<unsigned int, unsigned int>, QTreeWidgetItem*> ij_pairs;
};
