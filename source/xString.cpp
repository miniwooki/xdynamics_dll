#include "xstring.h"
//#include "xstringList.h"
//#include "xdynamics_algebra/xUtilityFunctions.h"

using namespace xdyn;

xstring::xstring()
	: wc(NULL)
	, len(0)
{

}

xstring::xstring(const char* _wc)
	: wc(NULL)
	, len(0)
{
	len = ((int)strnlen_s(_wc, sizeof(_wc)));
	if (len)
	{
		//wc = SysAllocStringLen(NULL, len);
		wc = new char[len];
		strcpy_s(wc, len, _wc);
	}	
}

xstring::xstring(const xstring& _xs)
	: len(_xs.size())
{
	if (len)
	{
		//wc = SysAllocStringLen(NULL, len);
		strcpy_s(wc, len, _xs.text());
	}
}

xstring::~xstring()
{
	delete[] wc; wc = NULL;// SysFreeString(wc);
}

void xstring::operator=(const xstring& _xs)
{
	len = _xs.size();
	if (wc)
		delete[] wc;
	if (len)
	{
		wc = new char[len];// wc = SysAllocStringLen(NULL, len);
	}
		
	strcpy_s(wc, len, _xs.text());
}

bool xstring::operator==(const char* _wc)
{
	return !strcmp(wc, _wc);
}

xstring xstring::operator+(const xstring& _ixs)
{
	int _len = len + _ixs.size();
	char *_wc = new char[len];// SysAllocStringLen(NULL, _len);
	strcpy_s(_wc, len, wc);
	strcat_s(_wc, _len, _ixs.text());
	xstring _xs(_wc);
	delete[] _wc;// SysFreeString(_wc);
 	return _xs;
}

xstring xstring::operator+(const char* _iwc)
{
	int _len = len + (int)strnlen_s(_iwc, sizeof(_iwc));
	char *_wc = new char[_len];// SysAllocStringLen(NULL, _len);
	strcpy_s(_wc, len, wc);
	strcat_s(_wc, _len, _iwc);
	xstring _xs(_wc);
	delete[] _wc;// SysFreeString(_wc);
	return _xs;
}

char* xstring::text() const
{
	return wc;
}

int xstring::size() const
{
	return len;
}

void xstring::split(const char* c, int n, int* data)
{
	//n = _n_split(c);
	//xstringList* slist = new xstringList;
	//_setupBuffer_int(n);
	string s = wc;
	basic_string<char>::size_type start = 0, end;
	static const basic_string<char>::size_type npos = -1;
	int len = (int)strnlen_s(c, sizeof(c));
	for (int i = 0; i < n; i++)
	{
		end = s.find(c, start);
		data[i] = atoi(s.substr(start, end - start).c_str());
		start = end + len;
	}
	
}

void xstring::split(const char* c, int n, double* data)
{
	
	//_setupBuffer_double(n);
	string s = wc;
	basic_string<char>::size_type start = 0, end;
	static const basic_string<char>::size_type npos = -1;
	int len = (int)strnlen_s(c, sizeof(c));
	for (int i = 0; i < n; i++)
	{
		end = s.find(c, start);
		data[i] = atof(s.substr(start, end - start).c_str());
		start = end + len;
	}
}

std::string xstring::toWString()
{
	std::string ws = wc;
	return ws;
}

const std::string xstring::toWString() const
{
	std::string ws = wc;
	return ws;
}

int xstring::_n_split(const char* c)
{
 	int n = 0;
// 	wstring s = wc;
// 	basic_string<char>::size_type start = 0, end;
// 	static const basic_string<char>::size_type npos = -1;
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

