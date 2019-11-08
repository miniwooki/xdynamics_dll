#ifndef XSTRING_H
#define XSTRING_H

#include "xdynamics_decl.h"
//#include <iostream>
#include <vector>

using namespace std;

class XDYNAMICS_API xstring
{
public:
	xstring();
	//xstring(std::string ws);
	xstring(const char* _wc);
	xstring(const xstring& _xs);
	xstring(const std::string& s);
	~xstring();

	void operator= (const xstring& _xs);
	bool operator== (const char* _wc);
	bool operator== (const xstring& _wc);
	bool operator!= (const xstring& _xs);
	xstring operator+ (const xstring& _xs);
	xstring operator+ (const char* _wc);
	xstring operator+= (const char* _wc);
	bool operator<(const xstring& _xs);
	bool operator>(const xstring& _xs);
	char* text() const;
	size_t size() const;
	int hexIndex();
	void split(const char* c, int n, int* data);
	void split(const char* c, int n, double* data);
	void split(const char* c, int n, std::string* data);
	unsigned int n_split_string(const char* c);
	std::string toStdString();
	const std::string toStdString() const;
	//static void get_split_data(xString* ls,  int* data);
	friend ostream& operator<<(ostream& os, const xstring& xs)
	{
		os << xs.text();
		return os;
	}
	//	friend ostream& operator<<(ostream& os, const xstring& xs);
private:
	std::vector<std::string> _n_split(const std::string& s, const char c);
	size_t len;
	char *wc;
};
xstring operator+ (const char* c, const xstring& s);

#endif