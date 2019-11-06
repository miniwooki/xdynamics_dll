#include "xstring.h"
#include <sstream>
#include <string>
#include <iostream>

//#include "xstringList.h"
//#include "xdynamics_algebra/xUtilityFunctions.h"

xstring::xstring()
	: wc(NULL)
	, len(0)
{

}

xstring::xstring(const char* _wc)
	: wc(NULL)
	, len(0)
{
	len = ((int)strnlen_s(_wc, 255)) + 1;
	if (len)
	{
		//wc = SysAllocStringLen(NULL, len);
		wc = new char[len];
		memset(wc, 0, sizeof(char) * len);
		strcpy_s(wc, len, _wc);
	}
}

xstring::xstring(const xstring& _xs)
	: len(_xs.size())
{
	if (len)
	{
		wc = new char[len];// SysAllocStringLen(NULL, len);
		memset(wc, 0, sizeof(char) * len);
		strcpy_s(wc, len, _xs.text());
	}
}

xstring::xstring(const std::string & s)
{
	const char* c = s.c_str();
	len = s.size() + 1;
	if (len)
	{
		wc = new char[len];// wc = SysAllocStringLen(NULL, len);
		memset(wc, 0, sizeof(char) * len);
		strcpy_s(wc, len, c);
	}		
	
}

xstring::~xstring()
{
	//
	if (wc)
	{
//#ifdef _DEBUG
//		std::cout << "delete - " << wc << std::endl;
//#endif
		delete[] wc; wc = NULL;// SysFreeString(wc);
	}
		
}

void xstring::operator=(const xstring& _xs)
{
	len = _xs.size();
	if (wc)
		delete[] wc;
	if (len)
	{
		wc = new char[len];// wc = SysAllocStringLen(NULL, len);
		strcpy_s(wc, len, _xs.text());
	}
}

bool xstring::operator==(const char* _wc)
{
	return !strcmp(wc, _wc);
}

bool xstring::operator==(const xstring& s)
{
	return !strcmp(wc, s.text());
}

bool xstring::operator!=(const xstring& _xs)
{
	return strcmp(wc, _xs.text());
}

xstring xstring::operator+=(const char* _wc)
{
	xstring o = *this + _wc;
	return o;
}

xstring operator+ (const char* c, const xstring& s)
{
	int _len = s.size() + (int)strnlen_s(c, 255);
	char *_wc = new char[_len + 1];// SysAllocStringLen(NULL, _len);
	memset(_wc, 0, _len);
	strcpy_s(_wc, _len, c);
	strcat_s(_wc, _len + 1, s.text());
	xstring _xs(_wc);
	delete[] _wc;// SysFreeString(_wc);
	return _xs;
}

xstring xstring::operator+(const xstring& _ixs)
{
	int _len = len + _ixs.size();
	char *_wc = new char[_len + 1];// SysAllocStringLen(NULL, _len);
	strcpy_s(_wc, _len, wc);
	strcat_s(_wc, _len + 1, _ixs.text());
	xstring _xs(_wc);
	delete[] _wc;// SysFreeString(_wc);
	return _xs;
}

xstring xstring::operator+(const char* _iwc)
{
	int _len = len + (int)strnlen_s(_iwc, 255);
	char *_wc = new char[_len + 1];// SysAllocStringLen(NULL, _len);
	memset(_wc, 0, _len);
	strcpy_s(_wc, _len, wc);
	strcat_s(_wc, _len + 1, _iwc);
	xstring _xs(_wc);
	delete[] _wc;// SysFreeString(_wc);
	return _xs;
}

char* xstring::text() const
{
	return wc;
}

size_t xstring::size() const
{
	return len;
}

void xstring::split(const char* c, int n, int* data)
{
	//std::string s = c;
	std::vector<std::string> d = _n_split(std::string(wc), *c);
	for (size_t i = 0; i < n; i++)
		data[i] = atoi(d.at(i).c_str());
}

void xstring::split(const char* c, int n, double* data)
{
	std::vector<std::string> d = _n_split(std::string(wc), *c);
	for (size_t i = 0; i < n; i++)
		data[i] = atof(d.at(i).c_str());
}

unsigned int xstring::n_split_string(const char * c)
{
	std::string input = this->toStdString();
	std::istringstream ss(input);
	std::string token;
	unsigned int cnt = 0;
	while (std::getline(ss, token, *c)) {
		cnt++;
	}
	return cnt;
}

void xstring::split(const char* c, int n, std::string* data)
{
	std::vector<std::string> d = _n_split(std::string(wc), *c);
	for (size_t i = 0; i < n; i++)
		data[i] = d.at(i);
}

std::string xstring::toStdString()
{
	std::string ws = wc;
	return ws;
}

const std::string xstring::toStdString() const
{
	std::string ws = wc;
	return ws;
}

std::vector<std::string> xstring::_n_split(const std::string& s, const char c)
{
	size_t start_pos = 0;
	size_t search_pos = 0;
	std::vector<std::string> result;
	while (start_pos < s.size())
	{
		search_pos = s.find_first_of(c, start_pos);
		std::string tmp_str;
		if (search_pos == std::string::npos)
		{
			search_pos = s.size();
			tmp_str = s.substr(start_pos, search_pos - start_pos);
			result.push_back(tmp_str);
			break;
		}
		tmp_str = s.substr(start_pos, search_pos - start_pos);
		if (!tmp_str.empty())
			result.push_back(tmp_str);
		start_pos = search_pos + 1;
	}
	return result;
}

bool xstring::operator<(const xstring& _xs)
{
	std::string a = text();
	std::string b = _xs.text();
	int c = a.compare(b);
	return c == -1 ? true : false;
}

bool xstring::operator>(const xstring& _xs)
{
	std::string a = text();
	std::string b = _xs.text();
	int c = a.compare(b);
	return c == -1 ? false : true;
}