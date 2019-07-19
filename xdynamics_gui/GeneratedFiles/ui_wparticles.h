/********************************************************************************
** Form generated from reading UI file 'wparticles.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WPARTICLES_H
#define UI_WPARTICLES_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wparticles
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QGroupBox *GBWParticles;
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QLabel *LName;
    QLineEdit *LEName;
    QLabel *LMaterial;
    QLineEdit *LEMaterial;
    QLabel *LNumThis;
    QLineEdit *LENumThis;
    QLabel *LNumTotal;
    QLineEdit *LENumTotal;
    QLabel *LMinRadius;
    QLineEdit *LEMinRadius;
    QLabel *LMaxRadius;
    QLineEdit *LEMaxRadius;

    void setupUi(QWidget *wparticles)
    {
        if (wparticles->objectName().isEmpty())
            wparticles->setObjectName(QString::fromUtf8("wparticles"));
        wparticles->resize(279, 160);
        wparticles->setMinimumSize(QSize(0, 160));
        wparticles->setMaximumSize(QSize(16777215, 160));
        gridLayout = new QGridLayout(wparticles);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wparticles);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        GBWParticles = new QGroupBox(frame);
        GBWParticles->setObjectName(QString::fromUtf8("GBWParticles"));
        gridLayout_4 = new QGridLayout(GBWParticles);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        LName = new QLabel(GBWParticles);
        LName->setObjectName(QString::fromUtf8("LName"));

        gridLayout_3->addWidget(LName, 0, 0, 1, 1);

        LEName = new QLineEdit(GBWParticles);
        LEName->setObjectName(QString::fromUtf8("LEName"));
        LEName->setReadOnly(true);

        gridLayout_3->addWidget(LEName, 0, 1, 1, 3);

        LMaterial = new QLabel(GBWParticles);
        LMaterial->setObjectName(QString::fromUtf8("LMaterial"));

        gridLayout_3->addWidget(LMaterial, 1, 0, 1, 1);

        LEMaterial = new QLineEdit(GBWParticles);
        LEMaterial->setObjectName(QString::fromUtf8("LEMaterial"));
        LEMaterial->setReadOnly(true);

        gridLayout_3->addWidget(LEMaterial, 1, 1, 1, 3);

        LNumThis = new QLabel(GBWParticles);
        LNumThis->setObjectName(QString::fromUtf8("LNumThis"));

        gridLayout_3->addWidget(LNumThis, 2, 0, 1, 1);

        LENumThis = new QLineEdit(GBWParticles);
        LENumThis->setObjectName(QString::fromUtf8("LENumThis"));
        LENumThis->setReadOnly(true);

        gridLayout_3->addWidget(LENumThis, 2, 1, 1, 1);

        LNumTotal = new QLabel(GBWParticles);
        LNumTotal->setObjectName(QString::fromUtf8("LNumTotal"));

        gridLayout_3->addWidget(LNumTotal, 2, 2, 1, 1);

        LENumTotal = new QLineEdit(GBWParticles);
        LENumTotal->setObjectName(QString::fromUtf8("LENumTotal"));
        LENumTotal->setReadOnly(true);

        gridLayout_3->addWidget(LENumTotal, 2, 3, 1, 1);

        LMinRadius = new QLabel(GBWParticles);
        LMinRadius->setObjectName(QString::fromUtf8("LMinRadius"));

        gridLayout_3->addWidget(LMinRadius, 3, 0, 1, 1);

        LEMinRadius = new QLineEdit(GBWParticles);
        LEMinRadius->setObjectName(QString::fromUtf8("LEMinRadius"));
        LEMinRadius->setReadOnly(true);

        gridLayout_3->addWidget(LEMinRadius, 3, 1, 1, 3);

        LMaxRadius = new QLabel(GBWParticles);
        LMaxRadius->setObjectName(QString::fromUtf8("LMaxRadius"));

        gridLayout_3->addWidget(LMaxRadius, 4, 0, 1, 1);

        LEMaxRadius = new QLineEdit(GBWParticles);
        LEMaxRadius->setObjectName(QString::fromUtf8("LEMaxRadius"));
        LEMaxRadius->setReadOnly(true);

        gridLayout_3->addWidget(LEMaxRadius, 4, 1, 1, 3);


        gridLayout_4->addLayout(gridLayout_3, 0, 0, 1, 1);


        gridLayout_2->addWidget(GBWParticles, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wparticles);

        QMetaObject::connectSlotsByName(wparticles);
    } // setupUi

    void retranslateUi(QWidget *wparticles)
    {
        wparticles->setWindowTitle(QApplication::translate("wparticles", "Form", nullptr));
        GBWParticles->setTitle(QApplication::translate("wparticles", "GroupBox", nullptr));
        LName->setText(QApplication::translate("wparticles", "Name", nullptr));
        LMaterial->setText(QApplication::translate("wparticles", "Material", nullptr));
        LNumThis->setText(QApplication::translate("wparticles", "This", nullptr));
        LNumTotal->setText(QApplication::translate("wparticles", "Total", nullptr));
        LMinRadius->setText(QApplication::translate("wparticles", "Min. radius", nullptr));
        LMaxRadius->setText(QApplication::translate("wparticles", "Max. radius", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wparticles: public Ui_wparticles {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WPARTICLES_H
