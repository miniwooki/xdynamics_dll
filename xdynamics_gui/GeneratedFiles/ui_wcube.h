/********************************************************************************
** Form generated from reading UI file 'wcubeQisQJO.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef WCUBEQISQJO_H
#define WCUBEQISQJO_H

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
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_wcube
{
public:
    QGridLayout *gridLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QGroupBox *GBGeometry;
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
    QLineEdit *LESZX;
    QLineEdit *LESZY;
    QLineEdit *LESZZ;

    void setupUi(QWidget *wcube)
    {
        if (wcube->objectName().isEmpty())
            wcube->setObjectName(QStringLiteral("wcube"));
        wcube->resize(256, 146);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(wcube->sizePolicy().hasHeightForWidth());
        wcube->setSizePolicy(sizePolicy);
        wcube->setMinimumSize(QSize(0, 146));
        wcube->setMaximumSize(QSize(16777215, 146));
        gridLayout = new QGridLayout(wcube);
        gridLayout->setSpacing(5);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        frame = new QFrame(wcube);
        frame->setObjectName(QStringLiteral("frame"));
        sizePolicy.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setSpacing(0);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        GBGeometry = new QGroupBox(frame);
        GBGeometry->setObjectName(QStringLiteral("GBGeometry"));
        sizePolicy.setHeightForWidth(GBGeometry->sizePolicy().hasHeightForWidth());
        GBGeometry->setSizePolicy(sizePolicy);
        GBGeometry->setMinimumSize(QSize(200, 144));
        GBGeometry->setMaximumSize(QSize(16777215, 200));
        gridLayout_4 = new QGridLayout(GBGeometry);
        gridLayout_4->setSpacing(5);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(5, 5, 5, 5);
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(5);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setSizeConstraint(QLayout::SetDefaultConstraint);
        gridLayout_3->setContentsMargins(9, 3, 9, 3);
        LName = new QLabel(GBGeometry);
        LName->setObjectName(QStringLiteral("LName"));

        gridLayout_3->addWidget(LName, 0, 0, 1, 1);

        LEName = new QLineEdit(GBGeometry);
        LEName->setObjectName(QStringLiteral("LEName"));
        LEName->setReadOnly(true);

        gridLayout_3->addWidget(LEName, 0, 1, 1, 3);

        LX = new QLabel(GBGeometry);
        LX->setObjectName(QStringLiteral("LX"));
        sizePolicy.setHeightForWidth(LX->sizePolicy().hasHeightForWidth());
        LX->setSizePolicy(sizePolicy);
        LX->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LX, 1, 1, 1, 1);

        LY = new QLabel(GBGeometry);
        LY->setObjectName(QStringLiteral("LY"));
        sizePolicy.setHeightForWidth(LY->sizePolicy().hasHeightForWidth());
        LY->setSizePolicy(sizePolicy);
        LY->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LY, 1, 2, 1, 1);

        LZ = new QLabel(GBGeometry);
        LZ->setObjectName(QStringLiteral("LZ"));
        sizePolicy.setHeightForWidth(LZ->sizePolicy().hasHeightForWidth());
        LZ->setSizePolicy(sizePolicy);
        LZ->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(LZ, 1, 3, 1, 1);

        LPoint1 = new QLabel(GBGeometry);
        LPoint1->setObjectName(QStringLiteral("LPoint1"));

        gridLayout_3->addWidget(LPoint1, 2, 0, 1, 1);

        LEP1X = new QLineEdit(GBGeometry);
        LEP1X->setObjectName(QStringLiteral("LEP1X"));
        LEP1X->setReadOnly(true);

        gridLayout_3->addWidget(LEP1X, 2, 1, 1, 1);

        LEP1Y = new QLineEdit(GBGeometry);
        LEP1Y->setObjectName(QStringLiteral("LEP1Y"));
        LEP1Y->setReadOnly(true);

        gridLayout_3->addWidget(LEP1Y, 2, 2, 1, 1);

        LEP1Z = new QLineEdit(GBGeometry);
        LEP1Z->setObjectName(QStringLiteral("LEP1Z"));
        LEP1Z->setReadOnly(true);

        gridLayout_3->addWidget(LEP1Z, 2, 3, 1, 1);

        LPoint2 = new QLabel(GBGeometry);
        LPoint2->setObjectName(QStringLiteral("LPoint2"));

        gridLayout_3->addWidget(LPoint2, 3, 0, 1, 1);

        LEP2X = new QLineEdit(GBGeometry);
        LEP2X->setObjectName(QStringLiteral("LEP2X"));
        LEP2X->setReadOnly(true);

        gridLayout_3->addWidget(LEP2X, 3, 1, 1, 1);

        LEP2Y = new QLineEdit(GBGeometry);
        LEP2Y->setObjectName(QStringLiteral("LEP2Y"));
        LEP2Y->setReadOnly(true);

        gridLayout_3->addWidget(LEP2Y, 3, 2, 1, 1);

        LEP2Z = new QLineEdit(GBGeometry);
        LEP2Z->setObjectName(QStringLiteral("LEP2Z"));
        LEP2Z->setReadOnly(true);

        gridLayout_3->addWidget(LEP2Z, 3, 3, 1, 1);

        LSize = new QLabel(GBGeometry);
        LSize->setObjectName(QStringLiteral("LSize"));

        gridLayout_3->addWidget(LSize, 4, 0, 1, 1);

        LESZX = new QLineEdit(GBGeometry);
        LESZX->setObjectName(QStringLiteral("LESZX"));
        LESZX->setReadOnly(true);

        gridLayout_3->addWidget(LESZX, 4, 1, 1, 1);

        LESZY = new QLineEdit(GBGeometry);
        LESZY->setObjectName(QStringLiteral("LESZY"));
        LESZY->setReadOnly(true);

        gridLayout_3->addWidget(LESZY, 4, 2, 1, 1);

        LESZZ = new QLineEdit(GBGeometry);
        LESZZ->setObjectName(QStringLiteral("LESZZ"));
        LESZZ->setReadOnly(true);

        gridLayout_3->addWidget(LESZZ, 4, 3, 1, 1);


        gridLayout_4->addLayout(gridLayout_3, 0, 0, 1, 1);


        gridLayout_2->addWidget(GBGeometry, 0, 0, 1, 1);


        gridLayout->addWidget(frame, 0, 0, 1, 1);


        retranslateUi(wcube);

        QMetaObject::connectSlotsByName(wcube);
    } // setupUi

    void retranslateUi(QWidget *wcube)
    {
        wcube->setWindowTitle(QApplication::translate("wcube", "Form", nullptr));
        GBGeometry->setTitle(QApplication::translate("wcube", "Geometry", nullptr));
        LName->setText(QApplication::translate("wcube", "Name", nullptr));
        LX->setText(QApplication::translate("wcube", "X", nullptr));
        LY->setText(QApplication::translate("wcube", "Y", nullptr));
        LZ->setText(QApplication::translate("wcube", "Z", nullptr));
        LPoint1->setText(QApplication::translate("wcube", "Point 1", nullptr));
        LPoint2->setText(QApplication::translate("wcube", "Point 2", nullptr));
        LSize->setText(QApplication::translate("wcube", "Size", nullptr));
    } // retranslateUi

};

namespace Ui {
    class wcube: public Ui_wcube {};
} // namespace Ui

QT_END_NAMESPACE

#endif // WCUBEQISQJO_H
