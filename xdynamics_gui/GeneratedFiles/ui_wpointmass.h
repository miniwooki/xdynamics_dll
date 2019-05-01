/********************************************************************************
** Form generated from reading UI file 'wpointmass.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WPOINTMASS_H
#define UI_WPOINTMASS_H

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
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wpointmass
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QGroupBox *GBBodyInformation;
    QGridLayout *gridLayout_5;
    QGridLayout *gridLayout_4;
    QLabel *LGeometry;
    QLineEdit *LEGeometry;
    QPushButton *PBConnectGeometry;
    QLabel *LName;
    QLineEdit *LEName;
    QLabel *LMass;
    QLineEdit *LEMass;
    QLabel *LPosition;
    QLineEdit *LEPosition;
    QLabel *LEulerParameters;
    QLineEdit *LEEuerParameters;
    QLabel *LEulerAngle;
    QLineEdit *LEEulerAngle;
    QGroupBox *GBInertia;
    QGridLayout *gridLayout_3;
    QLabel *LIxx;
    QLineEdit *LEIxx;
    QLabel *LIxy;
    QLineEdit *LEIxy;
    QLabel *LIyy;
    QLineEdit *LEIyy;
    QLabel *LIyz;
    QLineEdit *LEIyz;
    QLabel *LIzz;
    QLineEdit *LEIzz;
    QLabel *LIzx;
    QLineEdit *LEIzx;

    void setupUi(QWidget *wpointmass)
    {
        if (wpointmass->objectName().isEmpty())
            wpointmass->setObjectName(QStringLiteral("wpointmass"));
        wpointmass->resize(220, 300);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wpointmass->sizePolicy().hasHeightForWidth());
        wpointmass->setSizePolicy(sizePolicy);
        wpointmass->setMinimumSize(QSize(220, 300));
        wpointmass->setMaximumSize(QSize(16777215, 300));
        gridLayout = new QGridLayout(wpointmass);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wpointmass);
        frame->setObjectName(QStringLiteral("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setSpacing(0);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        GBBodyInformation = new QGroupBox(frame);
        GBBodyInformation->setObjectName(QStringLiteral("GBBodyInformation"));
        gridLayout_5 = new QGridLayout(GBBodyInformation);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(0, 0, 0, 0);
        gridLayout_4 = new QGridLayout();
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        LGeometry = new QLabel(GBBodyInformation);
        LGeometry->setObjectName(QStringLiteral("LGeometry"));

        gridLayout_4->addWidget(LGeometry, 0, 0, 1, 1);

        LEGeometry = new QLineEdit(GBBodyInformation);
        LEGeometry->setObjectName(QStringLiteral("LEGeometry"));

        gridLayout_4->addWidget(LEGeometry, 0, 1, 1, 1);

        PBConnectGeometry = new QPushButton(GBBodyInformation);
        PBConnectGeometry->setObjectName(QStringLiteral("PBConnectGeometry"));
        PBConnectGeometry->setMinimumSize(QSize(23, 23));
        PBConnectGeometry->setMaximumSize(QSize(23, 23));

        gridLayout_4->addWidget(PBConnectGeometry, 0, 2, 1, 1);

        LName = new QLabel(GBBodyInformation);
        LName->setObjectName(QStringLiteral("LName"));

        gridLayout_4->addWidget(LName, 1, 0, 1, 1);

        LEName = new QLineEdit(GBBodyInformation);
        LEName->setObjectName(QStringLiteral("LEName"));

        gridLayout_4->addWidget(LEName, 1, 1, 1, 2);

        LMass = new QLabel(GBBodyInformation);
        LMass->setObjectName(QStringLiteral("LMass"));

        gridLayout_4->addWidget(LMass, 2, 0, 1, 1);

        LEMass = new QLineEdit(GBBodyInformation);
        LEMass->setObjectName(QStringLiteral("LEMass"));

        gridLayout_4->addWidget(LEMass, 2, 1, 1, 2);

        LPosition = new QLabel(GBBodyInformation);
        LPosition->setObjectName(QStringLiteral("LPosition"));

        gridLayout_4->addWidget(LPosition, 3, 0, 1, 1);

        LEPosition = new QLineEdit(GBBodyInformation);
        LEPosition->setObjectName(QStringLiteral("LEPosition"));

        gridLayout_4->addWidget(LEPosition, 3, 1, 1, 2);

        LEulerParameters = new QLabel(GBBodyInformation);
        LEulerParameters->setObjectName(QStringLiteral("LEulerParameters"));

        gridLayout_4->addWidget(LEulerParameters, 4, 0, 1, 1);

        LEEuerParameters = new QLineEdit(GBBodyInformation);
        LEEuerParameters->setObjectName(QStringLiteral("LEEuerParameters"));

        gridLayout_4->addWidget(LEEuerParameters, 4, 1, 1, 2);

        LEulerAngle = new QLabel(GBBodyInformation);
        LEulerAngle->setObjectName(QStringLiteral("LEulerAngle"));

        gridLayout_4->addWidget(LEulerAngle, 5, 0, 1, 1);

        LEEulerAngle = new QLineEdit(GBBodyInformation);
        LEEulerAngle->setObjectName(QStringLiteral("LEEulerAngle"));

        gridLayout_4->addWidget(LEEulerAngle, 5, 1, 1, 2);

        GBInertia = new QGroupBox(GBBodyInformation);
        GBInertia->setObjectName(QStringLiteral("GBInertia"));
        gridLayout_3 = new QGridLayout(GBInertia);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setHorizontalSpacing(6);
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        LIxx = new QLabel(GBInertia);
        LIxx->setObjectName(QStringLiteral("LIxx"));

        gridLayout_3->addWidget(LIxx, 0, 0, 1, 1);

        LEIxx = new QLineEdit(GBInertia);
        LEIxx->setObjectName(QStringLiteral("LEIxx"));

        gridLayout_3->addWidget(LEIxx, 0, 1, 1, 1);

        LIxy = new QLabel(GBInertia);
        LIxy->setObjectName(QStringLiteral("LIxy"));

        gridLayout_3->addWidget(LIxy, 0, 2, 1, 1);

        LEIxy = new QLineEdit(GBInertia);
        LEIxy->setObjectName(QStringLiteral("LEIxy"));

        gridLayout_3->addWidget(LEIxy, 0, 3, 1, 1);

        LIyy = new QLabel(GBInertia);
        LIyy->setObjectName(QStringLiteral("LIyy"));

        gridLayout_3->addWidget(LIyy, 1, 0, 1, 1);

        LEIyy = new QLineEdit(GBInertia);
        LEIyy->setObjectName(QStringLiteral("LEIyy"));

        gridLayout_3->addWidget(LEIyy, 1, 1, 1, 1);

        LIyz = new QLabel(GBInertia);
        LIyz->setObjectName(QStringLiteral("LIyz"));

        gridLayout_3->addWidget(LIyz, 1, 2, 1, 1);

        LEIyz = new QLineEdit(GBInertia);
        LEIyz->setObjectName(QStringLiteral("LEIyz"));

        gridLayout_3->addWidget(LEIyz, 1, 3, 1, 1);

        LIzz = new QLabel(GBInertia);
        LIzz->setObjectName(QStringLiteral("LIzz"));

        gridLayout_3->addWidget(LIzz, 2, 0, 1, 1);

        LEIzz = new QLineEdit(GBInertia);
        LEIzz->setObjectName(QStringLiteral("LEIzz"));

        gridLayout_3->addWidget(LEIzz, 2, 1, 1, 1);

        LIzx = new QLabel(GBInertia);
        LIzx->setObjectName(QStringLiteral("LIzx"));

        gridLayout_3->addWidget(LIzx, 2, 2, 1, 1);

        LEIzx = new QLineEdit(GBInertia);
        LEIzx->setObjectName(QStringLiteral("LEIzx"));

        gridLayout_3->addWidget(LEIzx, 2, 3, 1, 1);


        gridLayout_4->addWidget(GBInertia, 6, 0, 1, 3);


        gridLayout_5->addLayout(gridLayout_4, 0, 0, 1, 1);


        gridLayout_2->addWidget(GBBodyInformation, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wpointmass);

        QMetaObject::connectSlotsByName(wpointmass);
    } // setupUi

    void retranslateUi(QWidget *wpointmass)
    {
        wpointmass->setWindowTitle(QApplication::translate("wpointmass", "Form", nullptr));
        GBBodyInformation->setTitle(QApplication::translate("wpointmass", "Body information", nullptr));
        LGeometry->setText(QApplication::translate("wpointmass", "Geometry", nullptr));
        LEGeometry->setText(QString());
        PBConnectGeometry->setText(QApplication::translate("wpointmass", "G", nullptr));
        LName->setText(QApplication::translate("wpointmass", "Name", nullptr));
        LMass->setText(QApplication::translate("wpointmass", "Mass", nullptr));
        LPosition->setText(QApplication::translate("wpointmass", "Position", nullptr));
        LEulerParameters->setText(QApplication::translate("wpointmass", "Euler parameters", nullptr));
        LEulerAngle->setText(QApplication::translate("wpointmass", "Euler Angle", nullptr));
        GBInertia->setTitle(QApplication::translate("wpointmass", "Moment of Inertia", nullptr));
        LIxx->setText(QApplication::translate("wpointmass", "Ixx", nullptr));
        LIxy->setText(QApplication::translate("wpointmass", "Ixy", nullptr));
        LIyy->setText(QApplication::translate("wpointmass", "Iyy", nullptr));
        LIyz->setText(QApplication::translate("wpointmass", "Iyz", nullptr));
        LIzz->setText(QApplication::translate("wpointmass", "Izz", nullptr));
        LIzx->setText(QApplication::translate("wpointmass", "Izx", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wpointmass: public Ui_wpointmass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WPOINTMASS_H
