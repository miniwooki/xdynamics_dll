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

public:
	xCheckCollisionDialog(QWidget* parent = nullptr);
	~xCheckCollisionDialog();

	void setup(xvParticle* _xp, xParticleManager* _xpmgr, xObjectManager* _omgr);

protected:
	void mouseReleaseEvent(QMouseEvent *event);

private slots:
	void checkCollision();
	void clickTreeItem(QTreeWidgetItem* item, int column);
	void highlightSelectedCluster(int row, int column);
	void selectedItemProcess();
	void changePosition(QTableWidgetItem* item);

//signals:
//	void highlightSelectedParticle(unsigned int id, QColor rgb, QColor&);

private:
	bool isSetup;
	bool isChangedSelection;
	xvParticle* xp;
	xObjectManager* omgr;
	xParticleManager* pmgr;
	QList<QListWidgetItem*> items;
	QMap<QString, xClusterInformation> cinfos;
	QMap<QString, QTreeWidgetItem*> parents;
	QMap<unsigned int, QColor> changedColor;
	//QMap<unsigned int, QTableWidgetItem*> tableItems;
	//QMap<QPair<unsigned int, unsigned int>, QTreeWidgetItem*> ij_pairs;
};
