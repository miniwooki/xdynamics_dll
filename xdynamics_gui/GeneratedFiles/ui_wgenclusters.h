/********************************************************************************
** Form generated from reading UI file 'wgenclusters.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WGENCLUSTERS_H
#define UI_WGENCLUSTERS_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wgenclusterparticles
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_3;
    QGroupBox *GB_GenerateParameters;
    QGridLayout *gridLayout_2;
    QLabel *LName;
    QLineEdit *LEName;
    QLabel *LX;
    QLabel *LY;
    QLabel *LZ;
    QLabel *LDim;
    QLineEdit *DX;
    QLineEdit *DY;
    QLineEdit *DZ;
    QLabel *LLoc;
    QLineEdit *LEY;
    QLineEdit *LEZ;
    QPushButton *PB_Gen;
    QLineEdit *LEX;
    QLabel *LScale;
    QSpinBox *SBScale;
    QPushButton *PBModify;
    QFrame *ViewFrame;

    void setupUi(QWidget *wgenclusterparticles)
    {
        if (wgenclusterparticles->objectName().isEmpty())
            wgenclusterparticles->setObjectName(QString::fromUtf8("wgenclusterparticles"));
        wgenclusterparticles->resize(264, 404);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wgenclusterparticles->sizePolicy().hasHeightForWidth());
        wgenclusterparticles->setSizePolicy(sizePolicy);
        wgenclusterparticles->setMinimumSize(QSize(0, 404));
        wgenclusterparticles->setMaximumSize(QSize(16777215, 404));
        gridLayout = new QGridLayout(wgenclusterparticles);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wgenclusterparticles);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(frame);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        GB_GenerateParameters = new QGroupBox(frame);
        GB_GenerateParameters->setObjectName(QString::fromUtf8("GB_GenerateParameters"));
        sizePolicy.setHeightForWidth(GB_GenerateParameters->sizePolicy().hasHeightForWidth());
        GB_GenerateParameters->setSizePolicy(sizePolicy);
        GB_GenerateParameters->setMinimumSize(QSize(0, 130));
        GB_GenerateParameters->setMaximumSize(QSize(16777215, 130));
        gridLayout_2 = new QGridLayout(GB_GenerateParameters);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setHorizontalSpacing(7);
        gridLayout_2->setVerticalSpacing(3);
        gridLayout_2->setContentsMargins(5, 3, 5, 7);
        LName = new QLabel(GB_GenerateParameters);
        LName->setObjectName(QString::fromUtf8("LName"));

        gridLayout_2->addWidget(LName, 0, 0, 1, 1);

        LEName = new QLineEdit(GB_GenerateParameters);
        LEName->setObjectName(QString::fromUtf8("LEName"));
        LEName->setReadOnly(true);

        gridLayout_2->addWidget(LEName, 0, 1, 1, 3);

        LX = new QLabel(GB_GenerateParameters);
        LX->setObjectName(QString::fromUtf8("LX"));
        LX->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(LX, 1, 1, 1, 1);

        LY = new QLabel(GB_GenerateParameters);
        LY->setObjectName(QString::fromUtf8("LY"));
        LY->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(LY, 1, 2, 1, 1);

        LZ = new QLabel(GB_GenerateParameters);
        LZ->setObjectName(QString::fromUtf8("LZ"));
        LZ->setAlignment(Qt::AlignCenter);

        gridLayout_2->addWidget(LZ, 1, 3, 1, 1);

        LDim = new QLabel(GB_GenerateParameters);
        LDim->setObjectName(QString::fromUtf8("LDim"));

        gridLayout_2->addWidget(LDim, 2, 0, 1, 1);

        DX = new QLineEdit(GB_GenerateParameters);
        DX->setObjectName(QString::fromUtf8("DX"));

        gridLayout_2->addWidget(DX, 2, 1, 1, 1);

        DY = new QLineEdit(GB_GenerateParameters);
        DY->setObjectName(QString::fromUtf8("DY"));

        gridLayout_2->addWidget(DY, 2, 2, 1, 1);

        DZ = new QLineEdit(GB_GenerateParameters);
        DZ->setObjectName(QString::fromUtf8("DZ"));

        gridLayout_2->addWidget(DZ, 2, 3, 1, 1);

        LLoc = new QLabel(GB_GenerateParameters);
        LLoc->setObjectName(QString::fromUtf8("LLoc"));

        gridLayout_2->addWidget(LLoc, 3, 0, 1, 1);

        LEY = new QLineEdit(GB_GenerateParameters);
        LEY->setObjectName(QString::fromUtf8("LEY"));

        gridLayout_2->addWidget(LEY, 3, 2, 1, 1);

        LEZ = new QLineEdit(GB_GenerateParameters);
        LEZ->setObjectName(QString::fromUtf8("LEZ"));

        gridLayout_2->addWidget(LEZ, 3, 3, 1, 1);

        PB_Gen = new QPushButton(GB_GenerateParameters);
        PB_Gen->setObjectName(QString::fromUtf8("PB_Gen"));
        PB_Gen->setMinimumSize(QSize(0, 25));
        PB_Gen->setMaximumSize(QSize(16777215, 25));

        gridLayout_2->addWidget(PB_Gen, 4, 1, 1, 3);

        LEX = new QLineEdit(GB_GenerateParameters);
        LEX->setObjectName(QString::fromUtf8("LEX"));

        gridLayout_2->addWidget(LEX, 3, 1, 1, 1);


        gridLayout_3->addWidget(GB_GenerateParameters, 0, 0, 1, 3);

        LScale = new QLabel(frame);
        LScale->setObjectName(QString::fromUtf8("LScale"));

        gridLayout_3->addWidget(LScale, 1, 0, 1, 1);

        SBScale = new QSpinBox(frame);
        SBScale->setObjectName(QString::fromUtf8("SBScale"));
        SBScale->setMinimumSize(QSize(124, 0));
        SBScale->setMaximumSize(QSize(124, 16777215));
        SBScale->setMinimum(1);
        SBScale->setMaximum(100);

        gridLayout_3->addWidget(SBScale, 1, 1, 1, 1);

        PBModify = new QPushButton(frame);
        PBModify->setObjectName(QString::fromUtf8("PBModify"));

        gridLayout_3->addWidget(PBModify, 1, 2, 1, 1);

        ViewFrame = new QFrame(frame);
        ViewFrame->setObjectName(QString::fromUtf8("ViewFrame"));
        ViewFrame->setMinimumSize(QSize(0, 240));
        ViewFrame->setMaximumSize(QSize(16777215, 240));
        ViewFrame->setFrameShape(QFrame::StyledPanel);
        ViewFrame->setFrameShadow(QFrame::Raised);

        gridLayout_3->addWidget(ViewFrame, 2, 0, 1, 3);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wgenclusterparticles);

        QMetaObject::connectSlotsByName(wgenclusterparticles);
    } // setupUi

    void retranslateUi(QWidget *wgenclusterparticles)
    {
        wgenclusterparticles->setWindowTitle(QApplication::translate("wgenclusterparticles", "Form", nullptr));
        GB_GenerateParameters->setTitle(QApplication::translate("wgenclusterparticles", "Generate parameters", nullptr));
        LName->setText(QApplication::translate("wgenclusterparticles", "Name", nullptr));
        LX->setText(QApplication::translate("wgenclusterparticles", "X", nullptr));
        LY->setText(QApplication::translate("wgenclusterparticles", "Y", nullptr));
        LZ->setText(QApplication::translate("wgenclusterparticles", "Z", nullptr));
        LDim->setText(QApplication::translate("wgenclusterparticles", "Dim.", nullptr));
        DX->setText(QApplication::translate("wgenclusterparticles", "1", nullptr));
        DY->setText(QApplication::translate("wgenclusterparticles", "1", nullptr));
        DZ->setText(QApplication::translate("wgenclusterparticles", "1", nullptr));
        LLoc->setText(QApplication::translate("wgenclusterparticles", "Loc.", nullptr));
        LEY->setText(QApplication::translate("wgenclusterparticles", "0.0", nullptr));
        LEZ->setText(QApplication::translate("wgenclusterparticles", "0.0", nullptr));
        PB_Gen->setText(QApplication::translate("wgenclusterparticles", "Generation", nullptr));
        LEX->setText(QApplication::translate("wgenclusterparticles", "0.0", nullptr));
        LScale->setText(QApplication::translate("wgenclusterparticles", "Scale", nullptr));
        PBModify->setText(QApplication::translate("wgenclusterparticles", "Modify", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wgenclusterparticles: public Ui_wgenclusterparticles {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WGENCLUSTERS_H
