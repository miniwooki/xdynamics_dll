/********************************************************************************
** Form generated from reading UI file 'wsimulation.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WSIMULATION_H
#define UI_WSIMULATION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
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
    QLabel *LMode;
    QLabel *LModeValue;
    QLabel *LNunParts;
    QLineEdit *LENumParts;
    QPushButton *PBSetting;

    void setupUi(QWidget *wsimulation)
    {
        if (wsimulation->objectName().isEmpty())
            wsimulation->setObjectName(QStringLiteral("wsimulation"));
        wsimulation->resize(265, 181);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wsimulation->sizePolicy().hasHeightForWidth());
        wsimulation->setSizePolicy(sizePolicy);
        wsimulation->setMinimumSize(QSize(0, 180));
        wsimulation->setMaximumSize(QSize(16777215, 181));
        gridLayout = new QGridLayout(wsimulation);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wsimulation);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_4 = new QGridLayout(frame);
        gridLayout_4->setSpacing(0);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        GBSimulationCondition = new QGroupBox(frame);
        GBSimulationCondition->setObjectName(QStringLiteral("GBSimulationCondition"));
        sizePolicy.setHeightForWidth(GBSimulationCondition->sizePolicy().hasHeightForWidth());
        GBSimulationCondition->setSizePolicy(sizePolicy);
        GBSimulationCondition->setMinimumSize(QSize(0, 102));
        GBSimulationCondition->setMaximumSize(QSize(16777215, 102));
        gridLayout_5 = new QGridLayout(GBSimulationCondition);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(6, 6, 6, 6);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setVerticalSpacing(6);
        LTimeStep = new QLabel(GBSimulationCondition);
        LTimeStep->setObjectName(QStringLiteral("LTimeStep"));

        gridLayout_3->addWidget(LTimeStep, 0, 0, 1, 1);

        LETimeStep = new QLineEdit(GBSimulationCondition);
        LETimeStep->setObjectName(QStringLiteral("LETimeStep"));

        gridLayout_3->addWidget(LETimeStep, 0, 1, 1, 1);

        LSaveStep = new QLabel(GBSimulationCondition);
        LSaveStep->setObjectName(QStringLiteral("LSaveStep"));

        gridLayout_3->addWidget(LSaveStep, 1, 0, 1, 1);

        LESaveStep = new QLineEdit(GBSimulationCondition);
        LESaveStep->setObjectName(QStringLiteral("LESaveStep"));

        gridLayout_3->addWidget(LESaveStep, 1, 1, 1, 1);

        LEndTime = new QLabel(GBSimulationCondition);
        LEndTime->setObjectName(QStringLiteral("LEndTime"));

        gridLayout_3->addWidget(LEndTime, 2, 0, 1, 1);

        LEEndTime = new QLineEdit(GBSimulationCondition);
        LEEndTime->setObjectName(QStringLiteral("LEEndTime"));

        gridLayout_3->addWidget(LEEndTime, 2, 1, 1, 1);


        gridLayout_5->addLayout(gridLayout_3, 0, 0, 1, 1);

        PBSolve = new QPushButton(GBSimulationCondition);
        PBSolve->setObjectName(QStringLiteral("PBSolve"));
        PBSolve->setMinimumSize(QSize(50, 76));
        PBSolve->setMaximumSize(QSize(50, 76));

        gridLayout_5->addWidget(PBSolve, 0, 1, 1, 1);


        verticalLayout->addWidget(GBSimulationCondition);

        GBSimulationInformation = new QGroupBox(frame);
        GBSimulationInformation->setObjectName(QStringLiteral("GBSimulationInformation"));
        sizePolicy.setHeightForWidth(GBSimulationInformation->sizePolicy().hasHeightForWidth());
        GBSimulationInformation->setSizePolicy(sizePolicy);
        GBSimulationInformation->setMinimumSize(QSize(0, 69));
        GBSimulationInformation->setMaximumSize(QSize(16777215, 69));
        gridLayout_6 = new QGridLayout(GBSimulationInformation);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        gridLayout_6->setContentsMargins(6, 6, 6, 6);
        gridLayout_2 = new QGridLayout();
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        LMode = new QLabel(GBSimulationInformation);
        LMode->setObjectName(QStringLiteral("LMode"));

        gridLayout_2->addWidget(LMode, 0, 0, 1, 1);

        LModeValue = new QLabel(GBSimulationInformation);
        LModeValue->setObjectName(QStringLiteral("LModeValue"));

        gridLayout_2->addWidget(LModeValue, 0, 1, 1, 2);

        LNunParts = new QLabel(GBSimulationInformation);
        LNunParts->setObjectName(QStringLiteral("LNunParts"));

        gridLayout_2->addWidget(LNunParts, 1, 0, 1, 2);

        LENumParts = new QLineEdit(GBSimulationInformation);
        LENumParts->setObjectName(QStringLiteral("LENumParts"));
        LENumParts->setReadOnly(true);

        gridLayout_2->addWidget(LENumParts, 1, 2, 1, 1);


        gridLayout_6->addLayout(gridLayout_2, 0, 0, 1, 1);

        PBSetting = new QPushButton(GBSimulationInformation);
        PBSetting->setObjectName(QStringLiteral("PBSetting"));
        PBSetting->setMinimumSize(QSize(50, 40));
        PBSetting->setMaximumSize(QSize(50, 40));

        gridLayout_6->addWidget(PBSetting, 0, 1, 1, 1);


        verticalLayout->addWidget(GBSimulationInformation);


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
        LMode->setText(QApplication::translate("wsimulation", "Mode", nullptr));
        LModeValue->setText(QApplication::translate("wsimulation", "None", nullptr));
        LNunParts->setText(QApplication::translate("wsimulation", "Num. parts", nullptr));
        PBSetting->setText(QApplication::translate("wsimulation", "Setting", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wsimulation: public Ui_wsimulation {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WSIMULATION_H
