; Listing generated by Microsoft (R) Optimizing Compiler Version 19.16.27034.0 

include listing.inc

INCLUDELIB MSVCRT
INCLUDELIB OLDNAMES

EXTRN	__imp_?staticMetaObject@QDockWidget@@2UQMetaObject@@B:BYTE
CONST	SEGMENT
?qt_meta_stringdata_xCommandWindow@@3Uqt_meta_stringdata_xCommandWindow_t@@B DD 0ffffffffH ; qt_meta_stringdata_xCommandWindow
	DD	0eH
	DD	00H
	ORG $+4
	DQ	0000000000000018H
	DB	078H
	DB	043H
	DB	06fH
	DB	06dH
	DB	06dH
	DB	061H
	DB	06eH
	DB	064H
	DB	057H
	DB	069H
	DB	06eH
	DB	064H
	DB	06fH
	DB	077H
	DB	00H
	ORG $+1
?qt_meta_data_xCommandWindow@@3QBIB DD 08H		; qt_meta_data_xCommandWindow
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
CONST	ENDS
PUBLIC	?metaObject@xCommandWindow@@UEBAPEBUQMetaObject@@XZ ; xCommandWindow::metaObject
PUBLIC	?qt_metacast@xCommandWindow@@UEAAPEAXPEBD@Z	; xCommandWindow::qt_metacast
PUBLIC	?qt_metacall@xCommandWindow@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z ; xCommandWindow::qt_metacall
PUBLIC	?qt_static_metacall@xCommandWindow@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z ; xCommandWindow::qt_static_metacall
PUBLIC	?staticMetaObject@xCommandWindow@@2UQMetaObject@@B ; xCommandWindow::staticMetaObject
EXTRN	__imp_?dynamicMetaObject@QObjectData@@QEBAPEAUQMetaObject@@XZ:PROC
EXTRN	__imp_?qt_metacast@QDockWidget@@UEAAPEAXPEBD@Z:PROC
EXTRN	__imp_?qt_metacall@QDockWidget@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z:PROC
CRT$XCU	SEGMENT
??staticMetaObject$initializer$@xCommandWindow@@2P6AXXZEA@@3P6AXXZEA DQ FLAT:??__E?staticMetaObject@xCommandWindow@@2UQMetaObject@@B@@YAXXZ ; ??staticMetaObject$initializer$@xCommandWindow@@2P6AXXZEA@@3P6AXXZEA
CRT$XCU	ENDS
_DATA	SEGMENT
?staticMetaObject@xCommandWindow@@2UQMetaObject@@B DB 8 DUP(00H) ; xCommandWindow::staticMetaObject
	DQ	FLAT:?qt_meta_stringdata_xCommandWindow@@3Uqt_meta_stringdata_xCommandWindow_t@@B
	DQ	FLAT:?qt_meta_data_xCommandWindow@@3QBIB
	DQ	FLAT:?qt_static_metacall@xCommandWindow@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z
	DQ	0000000000000000H
	DQ	0000000000000000H
_DATA	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ??__E?staticMetaObject@xCommandWindow@@2UQMetaObject@@B@@YAXXZ
text$di	SEGMENT
??__E?staticMetaObject@xCommandWindow@@2UQMetaObject@@B@@YAXXZ PROC ; `dynamic initializer for 'xCommandWindow::staticMetaObject'', COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xcommandwindow.cpp
; Line 66
	mov	rax, QWORD PTR __imp_?staticMetaObject@QDockWidget@@2UQMetaObject@@B
	mov	QWORD PTR ?staticMetaObject@xCommandWindow@@2UQMetaObject@@B, rax
; Line 71
	ret	0
??__E?staticMetaObject@xCommandWindow@@2UQMetaObject@@B@@YAXXZ ENDP ; `dynamic initializer for 'xCommandWindow::staticMetaObject''
text$di	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_static_metacall@xCommandWindow@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z
_TEXT	SEGMENT
_o$ = 8
_c$ = 16
_id$ = 24
_a$ = 32
?qt_static_metacall@xCommandWindow@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z PROC ; xCommandWindow::qt_static_metacall, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xcommandwindow.cpp
; Line 63
	ret	0
?qt_static_metacall@xCommandWindow@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z ENDP ; xCommandWindow::qt_static_metacall
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_metacall@xCommandWindow@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z
_TEXT	SEGMENT
this$ = 8
_c$ = 16
_id$ = 24
_a$ = 32
?qt_metacall@xCommandWindow@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z PROC ; xCommandWindow::qt_metacall, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xcommandwindow.cpp
; Line 90
	rex_jmp	QWORD PTR __imp_?qt_metacall@QDockWidget@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z
?qt_metacall@xCommandWindow@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z ENDP ; xCommandWindow::qt_metacall
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_metacast@xCommandWindow@@UEAAPEAXPEBD@Z
_TEXT	SEGMENT
this$ = 8
_clname$ = 16
?qt_metacast@xCommandWindow@@UEAAPEAXPEBD@Z PROC	; xCommandWindow::qt_metacast, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xcommandwindow.cpp
; Line 81
	mov	r9, rdx
	mov	r10, rcx
; Line 82
	test	rdx, rdx
	jne	SHORT $LN2@qt_metacas
	xor	eax, eax
; Line 86
	ret	0
$LN2@qt_metacas:
; Line 83
	lea	r8, OFFSET FLAT:?qt_meta_stringdata_xCommandWindow@@3Uqt_meta_stringdata_xCommandWindow_t@@B+24
	mov	rax, r9
	sub	r8, r9
	npad	5
$LL5@qt_metacas:
	movzx	edx, BYTE PTR [rax]
	movzx	ecx, BYTE PTR [rax+r8]
	sub	edx, ecx
	jne	SHORT $LN6@qt_metacas
	inc	rax
	test	ecx, ecx
	jne	SHORT $LL5@qt_metacas
$LN6@qt_metacas:
	test	edx, edx
	jne	SHORT $LN3@qt_metacas
; Line 84
	mov	rax, r10
; Line 86
	ret	0
$LN3@qt_metacas:
; Line 85
	mov	rdx, r9
	mov	rcx, r10
	rex_jmp	QWORD PTR __imp_?qt_metacast@QDockWidget@@UEAAPEAXPEBD@Z
?qt_metacast@xCommandWindow@@UEAAPEAXPEBD@Z ENDP	; xCommandWindow::qt_metacast
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?metaObject@xCommandWindow@@UEBAPEBUQMetaObject@@XZ
_TEXT	SEGMENT
this$ = 8
?metaObject@xCommandWindow@@UEBAPEBUQMetaObject@@XZ PROC ; xCommandWindow::metaObject, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xcommandwindow.cpp
; Line 77
	mov	rcx, QWORD PTR [rcx+8]
	cmp	QWORD PTR [rcx+40], 0
	je	SHORT $LN3@metaObject
	rex_jmp	QWORD PTR __imp_?dynamicMetaObject@QObjectData@@QEBAPEAUQMetaObject@@XZ
$LN3@metaObject:
	lea	rax, OFFSET FLAT:?staticMetaObject@xCommandWindow@@2UQMetaObject@@B ; xCommandWindow::staticMetaObject
; Line 78
	ret	0
?metaObject@xCommandWindow@@UEBAPEBUQMetaObject@@XZ ENDP ; xCommandWindow::metaObject
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ??C?$QScopedPointer@VQObjectData@@U?$QScopedPointerDeleter@VQObjectData@@@@@@QEBAPEAVQObjectData@@XZ
_TEXT	SEGMENT
this$ = 8
??C?$QScopedPointer@VQObjectData@@U?$QScopedPointerDeleter@VQObjectData@@@@@@QEBAPEAVQObjectData@@XZ PROC ; QScopedPointer<QObjectData,QScopedPointerDeleter<QObjectData> >::operator->, COMDAT
; File c:\qt\5.12.3\msvc2017_64\include\qtcore\qscopedpointer.h
; Line 118
	mov	rax, QWORD PTR [rcx]
; Line 119
	ret	0
??C?$QScopedPointer@VQObjectData@@U?$QScopedPointerDeleter@VQObjectData@@@@@@QEBAPEAVQObjectData@@XZ ENDP ; QScopedPointer<QObjectData,QScopedPointerDeleter<QObjectData> >::operator->
_TEXT	ENDS
END