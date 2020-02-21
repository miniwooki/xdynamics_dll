/********************************************************************************
** Form generated from reading UI file 'wsimulation.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WSIMULATION_H
#define UI_WSIMULATION_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wsimulation
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_4;
    QVBoxLayout *verticalLayout;
    QGroupBox *GBSimulationCondition;
    QGridLayout *gridLayout_5;
    QGridLayout *gridLayout_3;
    QLabel *LTimeStep;
    QLineEdit *LETimeStep;
    QLabel *LSaveStep;
    QLineEdit *LESaveStep;
    QLabel *LEndTime;
    QLineEdit *LEEndTime;
    QPushButton *PBSolve;
    QGroupBox *GBSimulationInformation;
    QGridLayout *gridLayout_6;
    QGridLayout *gridLayout_2;
    QLabel *LNumSteps;
    QLineEdit *LENumSteps;
    QLabel *LNunParts;
    QLineEdit *LENumParts;
    QPushButton *PBSetting;
    QGroupBox *GB_StartingPoint;
    QGridLayout *gridLayout_7;
    QPushButton *PB_Select_SP;
    QLineEdit *LE_StartingPoint;

    void setupUi(QWidget *wsimulation)
    {
        if (wsimulation->objectName().isEmpty())
            wsimulation->setObjectName(QString::fromUtf8("wsimulation"));
        wsimulation->resize(264, 262);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wsimulation->sizePolicy().hasHeightForWidth());
        wsimulation->setSizePolicy(sizePolicy);
        wsimulation->setMinimumSize(QSize(0, 262));
        wsimulation->setMaximumSize(QSize(16777215, 263));
        gridLayout = new QGridLayout(wsimulation);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wsimulation);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_4 = new QGridLayout(frame);
        gridLayout_4->setSpacing(0);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(10);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        GBSimulationCondition = new QGroupBox(frame);
        GBSimulationCondition->setObjectName(QString::fromUtf8("GBSimulationCondition"));
        sizePolicy.setHeightForWidth(GBSimulationCondition->sizePolicy().hasHeightForWidth());
        GBSimulationCondition->setSizePolicy(sizePolicy);
        GBSimulationCondition->setMinimumSize(QSize(0, 102));
        GBSimulationCondition->setMaximumSize(QSize(16777215, 102));
        gridLayout_5 = new QGridLayout(GBSimulationCondition);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_5->setVerticalSpacing(6);
        gridLayout_5->setContentsMargins(6, 6, 6, 6);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setVerticalSpacing(6);
        LTimeStep = new QLabel(GBSimulationCondition);
        LTimeStep->setObjectName(QString::fromUtf8("LTimeStep"));

        gridLayout_3->addWidget(LTimeStep, 0, 0, 1, 1);

        LETimeStep = new QLineEdit(GBSimulationCondition);
        LETimeStep->setObjectName(QString::fromUtf8("LETimeStep"));

        gridLayout_3->addWidget(LETimeStep, 0, 1, 1, 1);

        LSaveStep = new QLabel(GBSimulationCondition);
        LSaveStep->setObjectName(QString::fromUtf8("LSaveStep"));

        gridLayout_3->addWidget(LSaveStep, 1, 0, 1, 1);

        LESaveStep = new QLineEdit(GBSimulationCondition);
        LESaveStep->setObjectName(QString::fromUtf8("LESaveStep"));

        gridLayout_3->addWidget(LESaveStep, 1, 1, 1, 1);

        LEndTime = new QLabel(GBSimulationCondition);
        LEndTime->setObjectName(QString::fromUtf8("LEndTime"));

        gridLayout_3->addWidget(LEndTime, 2, 0, 1, 1);

        LEEndTime = new QLineEdit(GBSimulationCondition);
        LEEndTime->setObjectName(QString::fromUtf8("LEEndTime"));

        gridLayout_3->addWidget(LEEndTime, 2, 1, 1, 1);


        gridLayout_5->addLayout(gridLayout_3, 0, 0, 1, 1);

        PBSolve = new QPushButton(GBSimulationCondition);
        PBSolve->setObjectName(QString::fromUtf8("PBSolve"));
        PBSolve->setMinimumSize(QSize(50, 76));
        PBSolve->setMaximumSize(QSize(50, 76));

        gridLayout_5->addWidget(PBSolve, 0, 1, 1, 1);


        verticalLayout->addWidget(GBSimulationCondition);

        GBSimulationInformation = new QGroupBox(frame);
        GBSimulationInformation->setObjectName(QString::fromUtf8("GBSimulationInformation"));
        sizePolicy.setHeightForWidth(GBSimulationInformation->sizePolicy().hasHeightForWidth());
        GBSimulationInformation->setSizePolicy(sizePolicy);
        GBSimulationInformation->setMinimumSize(QSize(0, 80));
        GBSimulationInformation->setMaximumSize(QSize(16777215, 75));
        gridLayout_6 = new QGridLayout(GBSimulationInformation);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setVerticalSpacing(6);
        LNumSteps = new QLabel(GBSimulationInformation);
        LNumSteps->setObjectName(QString::fromUtf8("LNumSteps"));
        LNumSteps->setMinimumSize(QSize(0, 20));
        LNumSteps->setMaximumSize(QSize(16777215, 20));

        gridLayout_2->addWidget(LNumSteps, 0, 0, 1, 1);

        LENumSteps = new QLineEdit(GBSimulationInformation);
        LENumSteps->setObjectName(QString::fromUtf8("LENumSteps"));
        LENumSteps->setMinimumSize(QSize(0, 20));
        LENumSteps->setMaximumSize(QSize(16777215, 20));

        gridLayout_2->addWidget(LENumSteps, 0, 1, 1, 1);

        LNunParts = new QLabel(GBSimulationInformation);
        LNunParts->setObjectName(QString::fromUtf8("LNunParts"));
        LNunParts->setMinimumSize(QSize(0, 20));
        LNunParts->setMaximumSize(QSize(16777215, 20));

        gridLayout_2->addWidget(LNunParts, 1, 0, 1, 1);

        LENumParts = new QLineEdit(GBSimulationInformation);
        LENumParts->setObjectName(QString::fromUtf8("LENumParts"));
        LENumParts->setMinimumSize(QSize(0, 20));
        LENumParts->setMaximumSize(QSize(16777215, 20));
        LENumParts->setReadOnly(true);

        gridLayout_2->addWidget(LENumParts, 1, 1, 1, 1);


        gridLayout_6->addLayout(gridLayout_2, 0, 0, 1, 1);

        PBSetting = new QPushButton(GBSimulationInformation);
        PBSetting->setObjectName(QString::fromUtf8("PBSetting"));
        PBSetting->setMinimumSize(QSize(50, 45));
        PBSetting->setMaximumSize(QSize(50, 45));

        gridLayout_6->addWidget(PBSetting, 0, 1, 1, 1);


        verticalLayout->addWidget(GBSimulationInformation);

        GB_StartingPoint = new QGroupBox(frame);
        GB_StartingPoint->setObjectName(QString::fromUtf8("GB_StartingPoint"));
        GB_StartingPoint->setCheckable(true);
        GB_StartingPoint->setChecked(false);
        gridLayout_7 = new QGridLayout(GB_StartingPoint);
        gridLayout_7->setSpacing(0);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        gridLayout_7->setContentsMargins(0, 0, 0, 0);
        PB_Select_SP = new QPushButton(GB_StartingPoint);
        PB_Select_SP->setObjectName(QString::fromUtf8("PB_Select_SP"));

        gridLayout_7->addWidget(PB_Select_SP, 0, 0, 1, 1);

        LE_StartingPoint = new QLineEdit(GB_StartingPoint);
        LE_StartingPoint->setObjectName(QString::fromUtf8("LE_StartingPoint"));

        gridLayout_7->addWidget(LE_StartingPoint, 1, 0, 1, 1);


        verticalLayout->addWidget(GB_StartingPoint);


        gridLayout_4->addLayout(verticalLayout, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wsimulation);

        QMetaObject::connectSlotsByName(wsimulation);
    } // setupUi

    void retranslateUi(QWidget *wsimulation)
    {
        wsimulation->setWindowTitle(QApplication::translate("wsimulation", "Form", nullptr));
        GBSimulationCondition->setTitle(QApplication::translate("wsimulation", "Simulation condition", nullptr));
        LTimeStep->setText(QApplication::translate("wsimulation", "Time step", nullptr));
        LSaveStep->setText(QApplication::translate("wsimulation", "Save step", nullptr));
        LEndTime->setText(QApplication::translate("wsimulation", "End time", nullptr));
        PBSolve->setText(QApplication::translate("wsimulation", "Solve", nullptr));
        GBSimulationInformation->setTitle(QApplication::translate("wsimulation", "Simulation information", nullptr));
        LNumSteps->setText(QApplication::translate("wsimulation", "Num. steps", nullptr));
        LNunParts->setText(QApplication::translate("wsimulation", "Num. parts", nullptr));
        PBSetting->setText(QApplication::translate("wsimulation", "Setting", nullptr));
        GB_StartingPoint->setTitle(QApplication::translate("wsimulation", "Starting point", nullptr));
        PB_Select_SP->setText(QApplication::translate("wsimulation", "Select starting point", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wsimulation: public Ui_wsimulation {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WSIMULATION_H
