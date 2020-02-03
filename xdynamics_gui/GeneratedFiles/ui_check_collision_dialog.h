/********************************************************************************
** Form generated from reading UI file 'check_collision_dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CHECK_COLLISION_DIALOG_H
#define UI_CHECK_COLLISION_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTreeWidget>

QT_BEGIN_NAMESPACE

class Ui_CheckCollisionDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *L_ParticleList;
    QLabel *L_CollisionParticles;
    QLabel *L_Information;
    QListWidget *ParticleList;
    QTreeWidget *CollisionParticle;
    QTableWidget *Information;
    QPushButton *PB_Check;
    QPushButton *PB_Exit;

    void setupUi(QDialog *CheckCollisionDialog)
    {
        if (CheckCollisionDialog->objectName().isEmpty())
            CheckCollisionDialog->setObjectName(QString::fromUtf8("CheckCollisionDialog"));
        CheckCollisionDialog->resize(763, 401);
        gridLayout = new QGridLayout(CheckCollisionDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        L_ParticleList = new QLabel(CheckCollisionDialog);
        L_ParticleList->setObjectName(QString::fromUtf8("L_ParticleList"));

        gridLayout->addWidget(L_ParticleList, 0, 0, 1, 1);

        L_CollisionParticles = new QLabel(CheckCollisionDialog);
        L_CollisionParticles->setObjectName(QString::fromUtf8("L_CollisionParticles"));

        gridLayout->addWidget(L_CollisionParticles, 0, 1, 1, 1);

        L_Information = new QLabel(CheckCollisionDialog);
        L_Information->setObjectName(QString::fromUtf8("L_Information"));

        gridLayout->addWidget(L_Information, 0, 2, 1, 1);

        ParticleList = new QListWidget(CheckCollisionDialog);
        ParticleList->setObjectName(QString::fromUtf8("ParticleList"));
        ParticleList->setMinimumSize(QSize(150, 0));
        ParticleList->setMaximumSize(QSize(150, 16777215));

        gridLayout->addWidget(ParticleList, 1, 0, 1, 1);

        CollisionParticle = new QTreeWidget(CheckCollisionDialog);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QString::fromUtf8("1"));
        CollisionParticle->setHeaderItem(__qtreewidgetitem);
        CollisionParticle->setObjectName(QString::fromUtf8("CollisionParticle"));
        CollisionParticle->setMinimumSize(QSize(200, 0));
        CollisionParticle->setMaximumSize(QSize(200, 16777215));

        gridLayout->addWidget(CollisionParticle, 1, 1, 1, 1);

        Information = new QTableWidget(CheckCollisionDialog);
        Information->setObjectName(QString::fromUtf8("Information"));

        gridLayout->addWidget(Information, 1, 2, 1, 3);

        PB_Check = new QPushButton(CheckCollisionDialog);
        PB_Check->setObjectName(QString::fromUtf8("PB_Check"));

        gridLayout->addWidget(PB_Check, 2, 3, 1, 1);

        PB_Exit = new QPushButton(CheckCollisionDialog);
        PB_Exit->setObjectName(QString::fromUtf8("PB_Exit"));

        gridLayout->addWidget(PB_Exit, 2, 4, 1, 1);


        retranslateUi(CheckCollisionDialog);

        QMetaObject::connectSlotsByName(CheckCollisionDialog);
    } // setupUi

    void retranslateUi(QDialog *CheckCollisionDialog)
    {
        CheckCollisionDialog->setWindowTitle(QApplication::translate("CheckCollisionDialog", "Check Collision Dialog", nullptr));
        L_ParticleList->setText(QApplication::translate("CheckCollisionDialog", "Particles list", nullptr));
        L_CollisionParticles->setText(QApplication::translate("CheckCollisionDialog", "Collision particles", nullptr));
        L_Information->setText(QApplication::translate("CheckCollisionDialog", "Information of Clusters", nullptr));
        PB_Check->setText(QApplication::translate("CheckCollisionDialog", "Collision check", nullptr));
        PB_Exit->setText(QApplication::translate("CheckCollisionDialog", "Exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CheckCollisionDialog: public Ui_CheckCollisionDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CHECK_COLLISION_DIALOG_H
