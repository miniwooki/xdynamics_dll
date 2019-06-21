#include "xstring.h"
//#include "xstringList.h"
//#include "xdynamics_algebra/xUtilityFunctions.h"

using namespace xdyn;

xstring::xstring()
	: wc(NULL)
	, len(0)
{

}

xstring::xstring(const wchar_t* _wc)
	: wc(NULL)
	, len(0)
{
	len = ((int)wcslen(_wc)) * 2;
	if (len)
	{
		wc = SysAllocStringLen(NULL, len);
		wcscpy_s(wc, len, _wc);
	}	
}

xstring::xstring(const xstring& _xs)
	: len(_xs.size())
{
	if (len)
	{
		wc = SysAllocStringLen(NULL, len);
		wcscpy_s(wc, len, _xs.text());
	}
}

xstring::~xstring()
{
	SysFreeString(wc);
}

void xstring::operator=(const xstring& _xs)
{
	len = _xs.size();
	if (wc)
		SysFreeString(wc);
	if (len)
	{
		wc = SysAllocStringLen(NULL, len);
	}
		
	wcscpy_s(wc, len, _xs.text());
}

bool xstring::operator==(const wchar_t* _wc)
{
	return !wcscmp(wc, _wc);
}

xstring xstring::operator+(const xstring& _ixs)
{
	int _len = len + _ixs.size();
	wchar_t *_wc = SysAllocStringLen(NULL, _len);
	wcscpy_s(_wc, len, wc);
	wcscat_s(_wc, _len, _ixs.text());
	xstring _xs(_wc);
	SysFreeString(_wc);
 	return _xs;
}

xstring xstring::operator+(const wchar_t* _iwc)
{
	int _len = len + (int)wcslen(_iwc) * 2;
	wchar_t *_wc = SysAllocStringLen(NULL, _len);
	wcscpy_s(_wc, len, wc);
	wcscat_s(_wc, _len, _iwc);
	xstring _xs(_wc);
	SysFreeString(_wc);
	return _xs;
}

wchar_t* xstring::text() const
{
	return wc;
}

int xstring::size() const
{
	return len;
}

void xstring::split(const wchar_t* c, int n, int* data)
{
	//n = _n_split(c);
	//xstringList* slist = new xstringList;
	//_setupBuffer_int(n);
	wstring s = wc;
	basic_string<wchar_t>::size_type start = 0, end;
	static const basic_string<wchar_t>::size_type npos = -1;
	int len = (int)wcslen(c);
	for (int i = 0; i < n; i++)
	{
		end = s.find(c, start);
		data[i] = _wtoi(s.substr(start, end - start).c_str());
		start = end + len;
	}
	
}

void xstring::split(const wchar_t* c, int n, double* data)
{
	
	//_setupBuffer_double(n);
	wstring s = wc;
	basic_string<wchar_t>::size_type start = 0, end;
	static const basic_string<wchar_t>::size_type npos = -1;
	int len = (int)wcslen(c);
	for (int i = 0; i < n; i++)
	{
		end = s.find(c, start);
		data[i] = _wtof(s.substr(start, end - start).c_str());
		start = end + len;
	}
}

std::wstring xstring::toWString()
{
	std::wstring ws = wc;
	return ws;
}

const std::wstring xstring::toWString() const
{
	std::wstring ws = wc;
	return ws;
}

int xstring::_n_split(const wchar_t* c)
{
 	int n = 0;
// 	wstring s = wc;
// 	basic_string<wchar_t>::size_type start = 0, end;
// 	static const basic_string<wchar_t>::size_type npos = -1;
// 	int len = (int)wcslen(c);
// 	while (1)
// 	{
// 		end = s.find(c, start);
// 		start = end + len;
// 		n++;
// 		if (end == npos)
// 			break;
// 	}
 	return n;
}

