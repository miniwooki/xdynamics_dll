/****************************************************************************
** Meta object code from reading C++ file 'xparticle_result_dlg.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../include/xdynamics_gui/xparticle_result_dlg.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xparticle_result_dlg.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_xparticle_result_dlg_t {
    QByteArrayData data[5];
    char stringdata0[57];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xparticle_result_dlg_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xparticle_result_dlg_t qt_meta_stringdata_xparticle_result_dlg = {
    {
QT_MOC_LITERAL(0, 0, 20), // "xparticle_result_dlg"
QT_MOC_LITERAL(1, 21, 10), // "click_load"
QT_MOC_LITERAL(2, 32, 0), // ""
QT_MOC_LITERAL(3, 33, 12), // "click_export"
QT_MOC_LITERAL(4, 46, 10) // "click_exit"

    },
    "xparticle_result_dlg\0click_load\0\0"
    "click_export\0click_exit"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xparticle_result_dlg[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   29,    2, 0x08 /* Private */,
       3,    0,   30,    2, 0x08 /* Private */,
       4,    0,   31,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void xparticle_result_dlg::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<xparticle_result_dlg *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->click_load(); break;
        case 1: _t->click_export(); break;
        case 2: _t->click_exit(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject xparticle_result_dlg::staticMetaObject = { {
    &QDialog::staticMetaObject,
    qt_meta_stringdata_xparticle_result_dlg.data,
    qt_meta_data_xparticle_result_dlg,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *xparticle_result_dlg::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xparticle_result_dlg::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_xparticle_result_dlg.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int xparticle_result_dlg::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
