/****************************************************************************
** Meta object code from reading C++ file 'xCheckCollisionDialog.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xCheckCollisionDialog.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xCheckCollisionDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xCheckCollisionDialog_t {
    QByteArrayData data[19];
    char stringdata0[236];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xCheckCollisionDialog_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xCheckCollisionDialog_t qt_meta_stringdata_xCheckCollisionDialog = {
    {
QT_MOC_LITERAL(0, 0, 21), // "xCheckCollisionDialog"
QT_MOC_LITERAL(1, 22, 14), // "checkCollision"
QT_MOC_LITERAL(2, 37, 0), // ""
QT_MOC_LITERAL(3, 38, 11), // "clickCancel"
QT_MOC_LITERAL(4, 50, 13), // "clickTreeItem"
QT_MOC_LITERAL(5, 64, 16), // "QTreeWidgetItem*"
QT_MOC_LITERAL(6, 81, 4), // "item"
QT_MOC_LITERAL(7, 86, 6), // "column"
QT_MOC_LITERAL(8, 93, 24), // "highlightSelectedCluster"
QT_MOC_LITERAL(9, 118, 3), // "row"
QT_MOC_LITERAL(10, 122, 19), // "selectedItemProcess"
QT_MOC_LITERAL(11, 142, 9), // "movePlusX"
QT_MOC_LITERAL(12, 152, 9), // "movePlusY"
QT_MOC_LITERAL(13, 162, 9), // "movePlusZ"
QT_MOC_LITERAL(14, 172, 10), // "moveMinusX"
QT_MOC_LITERAL(15, 183, 10), // "moveMinusY"
QT_MOC_LITERAL(16, 194, 10), // "moveMinusZ"
QT_MOC_LITERAL(17, 205, 14), // "movePlusNormal"
QT_MOC_LITERAL(18, 220, 15) // "moveMinusNormal"

    },
    "xCheckCollisionDialog\0checkCollision\0"
    "\0clickCancel\0clickTreeItem\0QTreeWidgetItem*\0"
    "item\0column\0highlightSelectedCluster\0"
    "row\0selectedItemProcess\0movePlusX\0"
    "movePlusY\0movePlusZ\0moveMinusX\0"
    "moveMinusY\0moveMinusZ\0movePlusNormal\0"
    "moveMinusNormal"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xCheckCollisionDialog[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   79,    2, 0x08 /* Private */,
       3,    0,   80,    2, 0x08 /* Private */,
       4,    2,   81,    2, 0x08 /* Private */,
       8,    2,   86,    2, 0x08 /* Private */,
      10,    0,   91,    2, 0x08 /* Private */,
      11,    0,   92,    2, 0x08 /* Private */,
      12,    0,   93,    2, 0x08 /* Private */,
      13,    0,   94,    2, 0x08 /* Private */,
      14,    0,   95,    2, 0x08 /* Private */,
      15,    0,   96,    2, 0x08 /* Private */,
      16,    0,   97,    2, 0x08 /* Private */,
      17,    0,   98,    2, 0x08 /* Private */,
      18,    0,   99,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5, QMetaType::Int,    6,    7,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    9,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void xCheckCollisionDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<xCheckCollisionDialog *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->checkCollision(); break;
        case 1: _t->clickCancel(); break;
        case 2: _t->clickTreeItem((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: _t->highlightSelectedCluster((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->selectedItemProcess(); break;
        case 5: _t->movePlusX(); break;
        case 6: _t->movePlusY(); break;
        case 7: _t->movePlusZ(); break;
        case 8: _t->moveMinusX(); break;
        case 9: _t->moveMinusY(); break;
        case 10: _t->moveMinusZ(); break;
        case 11: _t->movePlusNormal(); break;
        case 12: _t->moveMinusNormal(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject xCheckCollisionDialog::staticMetaObject = { {
    &QDialog::staticMetaObject,
    qt_meta_stringdata_xCheckCollisionDialog.data,
    qt_meta_data_xCheckCollisionDialog,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *xCheckCollisionDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xCheckCollisionDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xCheckCollisionDialog.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "Ui::CheckCollisionDialog"))
        return static_cast< Ui::CheckCollisionDialog*>(this);
    return QDialog::qt_metacast(_clname);
}

int xCheckCollisionDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
