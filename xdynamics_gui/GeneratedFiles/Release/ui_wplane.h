/********************************************************************************
** Form generated from reading UI file 'wplane.ui'
**
** Created by: Qt User Interface Compiler version 5.12.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WPLANE_H
#define UI_WPLANE_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wplane
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QLabel *LName;
    QLineEdit *LEName;
    QLabel *LX;
    QLabel *LY;
    QLabel *LZ;
    QLabel *LPoint1;
    QLineEdit *LEP1X;
    QLineEdit *LEP1Y;
    QLineEdit *LEP1Z;
    QLabel *LPoint2;
    QLineEdit *LEP2X;
    QLineEdit *LEP2Y;
    QLineEdit *LEP2Z;
    QLabel *LSize;
    QLineEdit *LEDIRX;
    QLineEdit *LEDIRY;
    QLineEdit *LEDIRZ;

    void setupUi(QWidget *wplane)
    {
        if (wplane->objectName().isEmpty())
            wplane->setObjectName(QString::fromUtf8("wplane"));
        wplane->resize(256, 146);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wplane->sizePolicy().hasHeightForWidth());
        wplane->setSizePolicy(sizePolicy);
        wplane->setMinimumSize(QSize(0, 146));
        wplane->setMaximumSize(QSize(16777215, 146));
        gridLayout = new QGridLayout(wplane);
        gridLayout->setSpacing(0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wplane);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setSpacing(0);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        groupBox = new QGroupBox(frame);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout_4 = new QGridLayout(groupBox);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(6, 6, 6, 6);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(5);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setSizeConstraint(QLayout::SetDefaultConstraint);
        gridLayout_3->setContentsMargins(9, 3, 9, 3);
        LName = new QLabel(groupBox);
        LName->setObjectName(QString::fromUtf8("LName"));

        gridLayout_3->addWidget(LName, 0, 0, 1, 1);

        LEName = new QLineEdit(groupBox);
        LEName->setObjectName(QString::fromUtf8("LEName"));
        LEName->setReadOnly(true);

        gridLayout_3->addWidget(LEName, 0, 1, 1, 3);

        LX = new QLabel(groupBox);
        LX->setObjectName(QString::fromUtf8("LX"));
        sizePolicy.setHeightForWidth(LX->sizePolicy().hasHeightForWidth());
        LX->setSizePolicy(sizePolicy);
        LX->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LX, 1, 1, 1, 1);

        LY = new QLabel(groupBox);
        LY->setObjectName(QString::fromUtf8("LY"));
        sizePolicy.setHeightForWidth(LY->sizePolicy().hasHeightForWidth());
        LY->setSizePolicy(sizePolicy);
        LY->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LY, 1, 2, 1, 1);

        LZ = new QLabel(groupBox);
        LZ->setObjectName(QString::fromUtf8("LZ"));
        sizePolicy.setHeightForWidth(LZ->sizePolicy().hasHeightForWidth());
        LZ->setSizePolicy(sizePolicy);
        LZ->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LZ, 1, 3, 1, 1);

        LPoint1 = new QLabel(groupBox);
        LPoint1->setObjectName(QString::fromUtf8("LPoint1"));

        gridLayout_3->addWidget(LPoint1, 2, 0, 1, 1);

        LEP1X = new QLineEdit(groupBox);
        LEP1X->setObjectName(QString::fromUtf8("LEP1X"));
        LEP1X->setReadOnly(true);

        gridLayout_3->addWidget(LEP1X, 2, 1, 1, 1);

        LEP1Y = new QLineEdit(groupBox);
        LEP1Y->setObjectName(QString::fromUtf8("LEP1Y"));
        LEP1Y->setReadOnly(true);

        gridLayout_3->addWidget(LEP1Y, 2, 2, 1, 1);

        LEP1Z = new QLineEdit(groupBox);
        LEP1Z->setObjectName(QString::fromUtf8("LEP1Z"));
        LEP1Z->setReadOnly(true);

        gridLayout_3->addWidget(LEP1Z, 2, 3, 1, 1);

        LPoint2 = new QLabel(groupBox);
        LPoint2->setObjectName(QString::fromUtf8("LPoint2"));

        gridLayout_3->addWidget(LPoint2, 3, 0, 1, 1);

        LEP2X = new QLineEdit(groupBox);
        LEP2X->setObjectName(QString::fromUtf8("LEP2X"));
        LEP2X->setReadOnly(true);

        gridLayout_3->addWidget(LEP2X, 3, 1, 1, 1);

        LEP2Y = new QLineEdit(groupBox);
        LEP2Y->setObjectName(QString::fromUtf8("LEP2Y"));
        LEP2Y->setReadOnly(true);

        gridLayout_3->addWidget(LEP2Y, 3, 2, 1, 1);

        LEP2Z = new QLineEdit(groupBox);
        LEP2Z->setObjectName(QString::fromUtf8("LEP2Z"));
        LEP2Z->setReadOnly(true);

        gridLayout_3->addWidget(LEP2Z, 3, 3, 1, 1);

        LSize = new QLabel(groupBox);
        LSize->setObjectName(QString::fromUtf8("LSize"));

        gridLayout_3->addWidget(LSize, 4, 0, 1, 1);

        LEDIRX = new QLineEdit(groupBox);
        LEDIRX->setObjectName(QString::fromUtf8("LEDIRX"));
        LEDIRX->setReadOnly(true);

        gridLayout_3->addWidget(LEDIRX, 4, 1, 1, 1);

        LEDIRY = new QLineEdit(groupBox);
        LEDIRY->setObjectName(QString::fromUtf8("LEDIRY"));
        LEDIRY->setReadOnly(true);

        gridLayout_3->addWidget(LEDIRY, 4, 2, 1, 1);

        LEDIRZ = new QLineEdit(groupBox);
        LEDIRZ->setObjectName(QString::fromUtf8("LEDIRZ"));
        LEDIRZ->setReadOnly(true);

        gridLayout_3->addWidget(LEDIRZ, 4, 3, 1, 1);


        gridLayout_4->addLayout(gridLayout_3, 0, 0, 1, 1);


        gridLayout_2->addWidget(groupBox, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wplane);

        QMetaObject::connectSlotsByName(wplane);
    } // setupUi

    void retranslateUi(QWidget *wplane)
    {
        wplane->setWindowTitle(QApplication::translate("wplane", "Form", nullptr));
        groupBox->setTitle(QApplication::translate("wplane", "GroupBox", nullptr));
        LName->setText(QApplication::translate("wplane", "Name", nullptr));
        LX->setText(QApplication::translate("wplane", "X", nullptr));
        LY->setText(QApplication::translate("wplane", "Y", nullptr));
        LZ->setText(QApplication::translate("wplane", "Z", nullptr));
        LPoint1->setText(QApplication::translate("wplane", "Point 1", nullptr));
        LPoint2->setText(QApplication::translate("wplane", "Point 2", nullptr));
        LSize->setText(QApplication::translate("wplane", "Direction", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wplane: public Ui_wplane {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WPLANE_H
