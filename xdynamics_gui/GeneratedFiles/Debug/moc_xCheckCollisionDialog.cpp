/****************************************************************************
** Meta object code from reading C++ file 'xCheckCollisionDialog.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xCheckCollisionDialog.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xCheckCollisionDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xCheckCollisionDialog_t {
    QByteArrayData data[10];
    char stringdata0[130];
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
QT_MOC_LITERAL(3, 38, 13), // "clickTreeItem"
QT_MOC_LITERAL(4, 52, 16), // "QTreeWidgetItem*"
QT_MOC_LITERAL(5, 69, 4), // "item"
QT_MOC_LITERAL(6, 74, 6), // "column"
QT_MOC_LITERAL(7, 81, 24), // "highlightSelectedCluster"
QT_MOC_LITERAL(8, 106, 3), // "row"
QT_MOC_LITERAL(9, 110, 19) // "selectedItemProcess"

    },
    "xCheckCollisionDialog\0checkCollision\0"
    "\0clickTreeItem\0QTreeWidgetItem*\0item\0"
    "column\0highlightSelectedCluster\0row\0"
    "selectedItemProcess"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xCheckCollisionDialog[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x08 /* Private */,
       3,    2,   35,    2, 0x08 /* Private */,
       7,    2,   40,    2, 0x08 /* Private */,
       9,    0,   45,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4, QMetaType::Int,    5,    6,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    8,    6,
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
        case 1: _t->clickTreeItem((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: _t->highlightSelectedCluster((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: _t->selectedItemProcess(); break;
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
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
