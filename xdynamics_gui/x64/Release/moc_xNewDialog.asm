; Listing generated by Microsoft (R) Optimizing Compiler Version 19.16.27034.0 

include listing.inc

INCLUDELIB MSVCRT
INCLUDELIB OLDNAMES

EXTRN	__imp_?staticMetaObject@QDialog@@2UQMetaObject@@B:BYTE
CONST	SEGMENT
?qt_meta_stringdata_xNewDialog@@3Uqt_meta_stringdata_xNewDialog_t@@B DD 0ffffffffH ; qt_meta_stringdata_xNewDialog
	DD	0aH
	DD	00H
	ORG $+4
	DQ	0000000000000060H
	DD	0ffffffffH
	DD	08H
	DD	00H
	ORG $+4
	DQ	0000000000000053H
	DD	0ffffffffH
	DD	00H
	DD	00H
	ORG $+4
	DQ	0000000000000044H
	DD	0ffffffffH
	DD	0cH
	DD	00H
	ORG $+4
	DQ	000000000000002dH
	DB	078H
	DB	04eH
	DB	065H
	DB	077H
	DB	044H
	DB	069H
	DB	061H
	DB	06cH
	DB	06fH
	DB	067H
	DB	00H
	DB	043H
	DB	06cH
	DB	069H
	DB	063H
	DB	06bH
	DB	05fH
	DB	06fH
	DB	06bH
	DB	00H
	DB	00H
	DB	043H
	DB	06cH
	DB	069H
	DB	063H
	DB	06bH
	DB	05fH
	DB	062H
	DB	072H
	DB	06fH
	DB	077H
	DB	073H
	DB	065H
	DB	00H
	ORG $+6
	ORG $+8
?qt_meta_data_xNewDialog@@3QBIB DD 08H			; qt_meta_data_xNewDialog
	DD	00H
	DD	00H
	DD	00H
	DD	02H
	DD	0eH
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	00H
	DD	01H
	DD	00H
	DD	018H
	DD	02H
	DD	08H
	DD	03H
	DD	00H
	DD	019H
	DD	02H
	DD	08H
	DD	02bH
	DD	02bH
	DD	00H
CONST	ENDS
PUBLIC	?metaObject@xNewDialog@@UEBAPEBUQMetaObject@@XZ	; xNewDialog::metaObject
PUBLIC	?qt_metacast@xNewDialog@@UEAAPEAXPEBD@Z		; xNewDialog::qt_metacast
PUBLIC	?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z ; xNewDialog::qt_metacall
PUBLIC	?qt_static_metacall@xNewDialog@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z ; xNewDialog::qt_static_metacall
PUBLIC	?staticMetaObject@xNewDialog@@2UQMetaObject@@B	; xNewDialog::staticMetaObject
EXTRN	__imp_?dynamicMetaObject@QObjectData@@QEBAPEAUQMetaObject@@XZ:PROC
EXTRN	__imp_?qt_metacast@QDialog@@UEAAPEAXPEBD@Z:PROC
EXTRN	__imp_?qt_metacall@QDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z:PROC
EXTRN	?Click_ok@xNewDialog@@AEAAXXZ:PROC		; xNewDialog::Click_ok
EXTRN	?Click_browse@xNewDialog@@AEAAXXZ:PROC		; xNewDialog::Click_browse
;	COMDAT pdata
pdata	SEGMENT
$pdata$?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z DD imagerel $LN18
	DD	imagerel $LN18+125
	DD	imagerel $unwind$?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z
pdata	ENDS
CRT$XCU	SEGMENT
??staticMetaObject$initializer$@xNewDialog@@2P6AXXZEA@@3P6AXXZEA DQ FLAT:??__E?staticMetaObject@xNewDialog@@2UQMetaObject@@B@@YAXXZ ; ??staticMetaObject$initializer$@xNewDialog@@2P6AXXZEA@@3P6AXXZEA
CRT$XCU	ENDS
_DATA	SEGMENT
?staticMetaObject@xNewDialog@@2UQMetaObject@@B DB 8 DUP(00H) ; xNewDialog::staticMetaObject
	DQ	FLAT:?qt_meta_stringdata_xNewDialog@@3Uqt_meta_stringdata_xNewDialog_t@@B
	DQ	FLAT:?qt_meta_data_xNewDialog@@3QBIB
	DQ	FLAT:?qt_static_metacall@xNewDialog@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z
	DQ	0000000000000000H
	DQ	0000000000000000H
_DATA	ENDS
;	COMDAT xdata
xdata	SEGMENT
$unwind$?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z DD 081501H
	DD	087415H
	DD	076415H
	DD	063415H
	DD	0e0113215H
xdata	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ??__E?staticMetaObject@xNewDialog@@2UQMetaObject@@B@@YAXXZ
text$di	SEGMENT
??__E?staticMetaObject@xNewDialog@@2UQMetaObject@@B@@YAXXZ PROC ; `dynamic initializer for 'xNewDialog::staticMetaObject'', COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xnewdialog.cpp
; Line 83
	mov	rax, QWORD PTR __imp_?staticMetaObject@QDialog@@2UQMetaObject@@B
	mov	QWORD PTR ?staticMetaObject@xNewDialog@@2UQMetaObject@@B, rax
; Line 88
	ret	0
??__E?staticMetaObject@xNewDialog@@2UQMetaObject@@B@@YAXXZ ENDP ; `dynamic initializer for 'xNewDialog::staticMetaObject''
text$di	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_static_metacall@xNewDialog@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z
_TEXT	SEGMENT
_o$ = 8
_c$ = 16
_id$ = 24
_a$ = 32
?qt_static_metacall@xNewDialog@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z PROC ; xNewDialog::qt_static_metacall, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xnewdialog.cpp
; Line 70
	test	edx, edx
	jne	SHORT $LN7@qt_static_
; Line 73
	test	r8d, r8d
	je	SHORT $LN5@qt_static_
	cmp	r8d, 1
	jne	SHORT $LN7@qt_static_
; Line 75
	jmp	?Click_browse@xNewDialog@@AEAAXXZ	; xNewDialog::Click_browse
$LN5@qt_static_:
; Line 74
	jmp	?Click_ok@xNewDialog@@AEAAXXZ		; xNewDialog::Click_ok
$LN7@qt_static_:
; Line 80
	ret	0
?qt_static_metacall@xNewDialog@@CAXPEAVQObject@@W4Call@QMetaObject@@HPEAPEAX@Z ENDP ; xNewDialog::qt_static_metacall
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z
_TEXT	SEGMENT
this$ = 48
_c$ = 56
_id$ = 64
_a$ = 72
?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z PROC ; xNewDialog::qt_metacall, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xnewdialog.cpp
; Line 106
$LN18:
	mov	QWORD PTR [rsp+8], rbx
	mov	QWORD PTR [rsp+16], rsi
	mov	QWORD PTR [rsp+24], rdi
	push	r14
	sub	rsp, 32					; 00000020H
	mov	r14, r9
	mov	edi, edx
	mov	rsi, rcx
; Line 107
	call	QWORD PTR __imp_?qt_metacall@QDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z
	mov	ebx, eax
; Line 108
	test	eax, eax
	js	SHORT $LN1@qt_metacal
; Line 110
	test	edi, edi
	jne	SHORT $LN3@qt_metacal
; Line 111
	cmp	eax, 2
	jge	SHORT $LN7@qt_metacal
; Line 73
	test	eax, eax
	je	SHORT $LN13@qt_metacal
	cmp	eax, 1
	jne	SHORT $LN7@qt_metacal
; Line 75
	mov	rcx, rsi
	call	?Click_browse@xNewDialog@@AEAAXXZ	; xNewDialog::Click_browse
	jmp	SHORT $LN7@qt_metacal
$LN13@qt_metacal:
; Line 74
	mov	rcx, rsi
	call	?Click_ok@xNewDialog@@AEAAXXZ		; xNewDialog::Click_ok
; Line 113
	jmp	SHORT $LN7@qt_metacal
$LN3@qt_metacal:
; Line 114
	cmp	edi, 12
	jne	SHORT $LN6@qt_metacal
; Line 115
	cmp	ebx, 2
	jge	SHORT $LN7@qt_metacal
; Line 116
	mov	rax, QWORD PTR [r14]
	mov	DWORD PTR [rax], -1
$LN7@qt_metacal:
; Line 119
	sub	ebx, 2
$LN6@qt_metacal:
	mov	eax, ebx
$LN1@qt_metacal:
; Line 120
	mov	rbx, QWORD PTR [rsp+48]
	mov	rsi, QWORD PTR [rsp+56]
	mov	rdi, QWORD PTR [rsp+64]
	add	rsp, 32					; 00000020H
	pop	r14
	ret	0
?qt_metacall@xNewDialog@@UEAAHW4Call@QMetaObject@@HPEAPEAX@Z ENDP ; xNewDialog::qt_metacall
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?qt_metacast@xNewDialog@@UEAAPEAXPEBD@Z
_TEXT	SEGMENT
this$ = 8
_clname$ = 16
?qt_metacast@xNewDialog@@UEAAPEAXPEBD@Z PROC		; xNewDialog::qt_metacast, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xnewdialog.cpp
; Line 98
	mov	r9, rdx
	mov	r10, rcx
; Line 99
	test	rdx, rdx
	jne	SHORT $LN2@qt_metacas
	xor	eax, eax
; Line 103
	ret	0
$LN2@qt_metacas:
; Line 100
	lea	r8, OFFSET FLAT:?qt_meta_stringdata_xNewDialog@@3Uqt_meta_stringdata_xNewDialog_t@@B+96
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
; Line 101
	mov	rax, r10
; Line 103
	ret	0
$LN3@qt_metacas:
; Line 102
	mov	rdx, r9
	mov	rcx, r10
	rex_jmp	QWORD PTR __imp_?qt_metacast@QDialog@@UEAAPEAXPEBD@Z
?qt_metacast@xNewDialog@@UEAAPEAXPEBD@Z ENDP		; xNewDialog::qt_metacast
_TEXT	ENDS
; Function compile flags: /Ogtpy
;	COMDAT ?metaObject@xNewDialog@@UEBAPEBUQMetaObject@@XZ
_TEXT	SEGMENT
this$ = 8
?metaObject@xNewDialog@@UEBAPEBUQMetaObject@@XZ PROC	; xNewDialog::metaObject, COMDAT
; File c:\xdynamics\xdynamics_gui\generatedfiles\release\moc_xnewdialog.cpp
; Line 94
	mov	rcx, QWORD PTR [rcx+8]
	cmp	QWORD PTR [rcx+40], 0
	je	SHORT $LN3@metaObject
	rex_jmp	QWORD PTR __imp_?dynamicMetaObject@QObjectData@@QEBAPEAUQMetaObject@@XZ
$LN3@metaObject:
	lea	rax, OFFSET FLAT:?staticMetaObject@xNewDialog@@2UQMetaObject@@B ; xNewDialog::staticMetaObject
; Line 95
	ret	0
?metaObject@xNewDialog@@UEBAPEBUQMetaObject@@XZ ENDP	; xNewDialog::metaObject
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