/****************************************************************************
** Meta object code from reading C++ file 'gen_cluster_dlg.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/gen_cluster_dlg.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gen_cluster_dlg.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ScatterDataModifier_t {
    QByteArrayData data[1];
    char stringdata0[20];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ScatterDataModifier_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ScatterDataModifier_t qt_meta_stringdata_ScatterDataModifier = {
    {
QT_MOC_LITERAL(0, 0, 19) // "ScatterDataModifier"

    },
    "ScatterDataModifier"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ScatterDataModifier[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void ScatterDataModifier::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject ScatterDataModifier::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_ScatterDataModifier.data,
    qt_meta_data_ScatterDataModifier,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ScatterDataModifier::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ScatterDataModifier::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ScatterDataModifier.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int ScatterDataModifier::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_gen_cluster_dlg_t {
    QByteArrayData data[10];
    char stringdata0[113];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_gen_cluster_dlg_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_gen_cluster_dlg_t qt_meta_stringdata_gen_cluster_dlg = {
    {
QT_MOC_LITERAL(0, 0, 15), // "gen_cluster_dlg"
QT_MOC_LITERAL(1, 16, 8), // "clickAdd"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 10), // "clickApply"
QT_MOC_LITERAL(4, 37, 11), // "clickCancel"
QT_MOC_LITERAL(5, 49, 9), // "clickCell"
QT_MOC_LITERAL(6, 59, 10), // "changeItem"
QT_MOC_LITERAL(7, 70, 17), // "QTableWidgetItem*"
QT_MOC_LITERAL(8, 88, 12), // "increaseRows"
QT_MOC_LITERAL(9, 101, 11) // "changeScale"

    },
    "gen_cluster_dlg\0clickAdd\0\0clickApply\0"
    "clickCancel\0clickCell\0changeItem\0"
    "QTableWidgetItem*\0increaseRows\0"
    "changeScale"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_gen_cluster_dlg[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x08 /* Private */,
       3,    0,   50,    2, 0x08 /* Private */,
       4,    0,   51,    2, 0x08 /* Private */,
       5,    2,   52,    2, 0x08 /* Private */,
       6,    1,   57,    2, 0x08 /* Private */,
       8,    1,   60,    2, 0x08 /* Private */,
       9,    1,   63,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    2,    2,
    QMetaType::Void, 0x80000000 | 7,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void, QMetaType::Int,    2,

       0        // eod
};

void gen_cluster_dlg::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<gen_cluster_dlg *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->clickAdd(); break;
        case 1: _t->clickApply(); break;
        case 2: _t->clickCancel(); break;
        case 3: _t->clickCell((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->changeItem((*reinterpret_cast< QTableWidgetItem*(*)>(_a[1]))); break;
        case 5: _t->increaseRows((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->changeScale((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject gen_cluster_dlg::staticMetaObject = { {
    &QDialog::staticMetaObject,
    qt_meta_stringdata_gen_cluster_dlg.data,
    qt_meta_data_gen_cluster_dlg,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *gen_cluster_dlg::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *gen_cluster_dlg::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_gen_cluster_dlg.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int gen_cluster_dlg::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
