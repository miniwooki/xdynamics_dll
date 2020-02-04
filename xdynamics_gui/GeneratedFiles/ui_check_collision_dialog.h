/********************************************************************************
** Form generated from reading UI file 'check_collision_dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
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
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTreeWidget>

QT_BEGIN_NAMESPACE

class Ui_CheckCollisionDialog
{
public:
    QGridLayout *gridLayout_2;
    QLabel *L_ParticleList;
    QLabel *L_CollisionParticles;
    QListWidget *ParticleList;
    QTreeWidget *CollisionParticle;
    QGridLayout *gridLayout;
    QLabel *L_Overlap;
    QLineEdit *LE_Overlap;
    QLabel *L_Direction;
    QLineEdit *LE_Direction;
    QLabel *L_MoveLength;
    QLineEdit *LE_MoveLength;
    QPushButton *MinusX;
    QPushButton *PlusX;
    QPushButton *MinusY;
    QPushButton *PlusY;
    QPushButton *MinusZ;
    QPushButton *PlusZ;
    QPushButton *MinusNormal;
    QPushButton *PlusNormal;
    QPushButton *PB_Check;
    QPushButton *PB_Exit;

    void setupUi(QDialog *CheckCollisionDialog)
    {
        if (CheckCollisionDialog->objectName().isEmpty())
            CheckCollisionDialog->setObjectName(QString::fromUtf8("CheckCollisionDialog"));
        CheckCollisionDialog->resize(525, 433);
        CheckCollisionDialog->setMinimumSize(QSize(525, 433));
        CheckCollisionDialog->setMaximumSize(QSize(525, 433));
        gridLayout_2 = new QGridLayout(CheckCollisionDialog);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(5, 5, 5, 5);
        L_ParticleList = new QLabel(CheckCollisionDialog);
        L_ParticleList->setObjectName(QString::fromUtf8("L_ParticleList"));

        gridLayout_2->addWidget(L_ParticleList, 0, 0, 1, 1);

        L_CollisionParticles = new QLabel(CheckCollisionDialog);
        L_CollisionParticles->setObjectName(QString::fromUtf8("L_CollisionParticles"));

        gridLayout_2->addWidget(L_CollisionParticles, 0, 1, 1, 1);

        ParticleList = new QListWidget(CheckCollisionDialog);
        ParticleList->setObjectName(QString::fromUtf8("ParticleList"));
        ParticleList->setMinimumSize(QSize(150, 0));
        ParticleList->setMaximumSize(QSize(150, 16777215));

        gridLayout_2->addWidget(ParticleList, 1, 0, 1, 1);

        CollisionParticle = new QTreeWidget(CheckCollisionDialog);
        QTreeWidgetItem *__qtreewidgetitem = new QTreeWidgetItem();
        __qtreewidgetitem->setText(0, QString::fromUtf8("1"));
        CollisionParticle->setHeaderItem(__qtreewidgetitem);
        CollisionParticle->setObjectName(QString::fromUtf8("CollisionParticle"));
        CollisionParticle->setMinimumSize(QSize(200, 0));
        CollisionParticle->setMaximumSize(QSize(200, 16777215));

        gridLayout_2->addWidget(CollisionParticle, 1, 1, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        L_Overlap = new QLabel(CheckCollisionDialog);
        L_Overlap->setObjectName(QString::fromUtf8("L_Overlap"));

        gridLayout->addWidget(L_Overlap, 0, 0, 1, 2);

        LE_Overlap = new QLineEdit(CheckCollisionDialog);
        LE_Overlap->setObjectName(QString::fromUtf8("LE_Overlap"));

        gridLayout->addWidget(LE_Overlap, 1, 0, 1, 2);

        L_Direction = new QLabel(CheckCollisionDialog);
        L_Direction->setObjectName(QString::fromUtf8("L_Direction"));

        gridLayout->addWidget(L_Direction, 2, 0, 1, 2);

        LE_Direction = new QLineEdit(CheckCollisionDialog);
        LE_Direction->setObjectName(QString::fromUtf8("LE_Direction"));

        gridLayout->addWidget(LE_Direction, 3, 0, 1, 2);

        L_MoveLength = new QLabel(CheckCollisionDialog);
        L_MoveLength->setObjectName(QString::fromUtf8("L_MoveLength"));

        gridLayout->addWidget(L_MoveLength, 4, 0, 1, 1);

        LE_MoveLength = new QLineEdit(CheckCollisionDialog);
        LE_MoveLength->setObjectName(QString::fromUtf8("LE_MoveLength"));

        gridLayout->addWidget(LE_MoveLength, 5, 0, 1, 2);

        MinusX = new QPushButton(CheckCollisionDialog);
        MinusX->setObjectName(QString::fromUtf8("MinusX"));
        MinusX->setMinimumSize(QSize(70, 50));
        MinusX->setMaximumSize(QSize(70, 50));
        QFont font;
        font.setPointSize(12);
        MinusX->setFont(font);

        gridLayout->addWidget(MinusX, 6, 0, 1, 1);

        PlusX = new QPushButton(CheckCollisionDialog);
        PlusX->setObjectName(QString::fromUtf8("PlusX"));
        PlusX->setMinimumSize(QSize(70, 50));
        PlusX->setMaximumSize(QSize(70, 50));
        PlusX->setFont(font);

        gridLayout->addWidget(PlusX, 6, 1, 1, 1);

        MinusY = new QPushButton(CheckCollisionDialog);
        MinusY->setObjectName(QString::fromUtf8("MinusY"));
        MinusY->setMinimumSize(QSize(70, 50));
        MinusY->setMaximumSize(QSize(70, 50));
        MinusY->setFont(font);

        gridLayout->addWidget(MinusY, 7, 0, 1, 1);

        PlusY = new QPushButton(CheckCollisionDialog);
        PlusY->setObjectName(QString::fromUtf8("PlusY"));
        PlusY->setMinimumSize(QSize(70, 50));
        PlusY->setMaximumSize(QSize(70, 50));
        PlusY->setFont(font);

        gridLayout->addWidget(PlusY, 7, 1, 1, 1);

        MinusZ = new QPushButton(CheckCollisionDialog);
        MinusZ->setObjectName(QString::fromUtf8("MinusZ"));
        MinusZ->setMinimumSize(QSize(70, 50));
        MinusZ->setMaximumSize(QSize(70, 50));
        MinusZ->setFont(font);

        gridLayout->addWidget(MinusZ, 8, 0, 1, 1);

        PlusZ = new QPushButton(CheckCollisionDialog);
        PlusZ->setObjectName(QString::fromUtf8("PlusZ"));
        PlusZ->setMinimumSize(QSize(70, 50));
        PlusZ->setMaximumSize(QSize(70, 50));
        PlusZ->setFont(font);

        gridLayout->addWidget(PlusZ, 8, 1, 1, 1);

        MinusNormal = new QPushButton(CheckCollisionDialog);
        MinusNormal->setObjectName(QString::fromUtf8("MinusNormal"));
        MinusNormal->setMinimumSize(QSize(70, 30));
        MinusNormal->setMaximumSize(QSize(70, 30));

        gridLayout->addWidget(MinusNormal, 9, 0, 1, 1);

        PlusNormal = new QPushButton(CheckCollisionDialog);
        PlusNormal->setObjectName(QString::fromUtf8("PlusNormal"));
        PlusNormal->setMinimumSize(QSize(70, 30));
        PlusNormal->setMaximumSize(QSize(70, 30));

        gridLayout->addWidget(PlusNormal, 9, 1, 1, 1);

        PB_Check = new QPushButton(CheckCollisionDialog);
        PB_Check->setObjectName(QString::fromUtf8("PB_Check"));
        PB_Check->setMinimumSize(QSize(148, 30));
        PB_Check->setMaximumSize(QSize(148, 30));

        gridLayout->addWidget(PB_Check, 10, 0, 1, 2);

        PB_Exit = new QPushButton(CheckCollisionDialog);
        PB_Exit->setObjectName(QString::fromUtf8("PB_Exit"));
        PB_Exit->setMinimumSize(QSize(148, 30));
        PB_Exit->setMaximumSize(QSize(148, 30));

        gridLayout->addWidget(PB_Exit, 11, 0, 1, 2);


        gridLayout_2->addLayout(gridLayout, 1, 2, 1, 1);


        retranslateUi(CheckCollisionDialog);

        QMetaObject::connectSlotsByName(CheckCollisionDialog);
    } // setupUi

    void retranslateUi(QDialog *CheckCollisionDialog)
    {
        CheckCollisionDialog->setWindowTitle(QApplication::translate("CheckCollisionDialog", "Check Collision Dialog", nullptr));
        L_ParticleList->setText(QApplication::translate("CheckCollisionDialog", "Particles list", nullptr));
        L_CollisionParticles->setText(QApplication::translate("CheckCollisionDialog", "Collision particles", nullptr));
        L_Overlap->setText(QApplication::translate("CheckCollisionDialog", "Collision overlap", nullptr));
        L_Direction->setText(QApplication::translate("CheckCollisionDialog", "Collision direction", nullptr));
        L_MoveLength->setText(QApplication::translate("CheckCollisionDialog", "Move length", nullptr));
        MinusX->setText(QApplication::translate("CheckCollisionDialog", "-X", nullptr));
        PlusX->setText(QApplication::translate("CheckCollisionDialog", "+X", nullptr));
        MinusY->setText(QApplication::translate("CheckCollisionDialog", "-Y", nullptr));
        PlusY->setText(QApplication::translate("CheckCollisionDialog", "+Y", nullptr));
        MinusZ->setText(QApplication::translate("CheckCollisionDialog", "-Z", nullptr));
        PlusZ->setText(QApplication::translate("CheckCollisionDialog", "+Z", nullptr));
        MinusNormal->setText(QApplication::translate("CheckCollisionDialog", "- Normal", nullptr));
        PlusNormal->setText(QApplication::translate("CheckCollisionDialog", "+ Normal", nullptr));
        PB_Check->setText(QApplication::translate("CheckCollisionDialog", "Collision check", nullptr));
        PB_Exit->setText(QApplication::translate("CheckCollisionDialog", "Exit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CheckCollisionDialog: public Ui_CheckCollisionDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CHECK_COLLISION_DIALOG_H
