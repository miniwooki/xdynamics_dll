#ifndef XSTRING_H
#define XSTRING_H

#include "xdynamics_decl.h"
//#include "xStringList.h"

//class xStringList;
namespace xdyn
{
	class XDYNAMICS_API xstring
	{
	public:
		xstring();
		//xstring(std::string ws);
		xstring(const char* _wc);
		xstring(const xstring& _xs);
		~xstring();

		void operator= (const xstring& _xs);
		bool operator== (const char* _wc);
		xstring operator+ (const xstring& _xs);
		xstring operator+ (const char* _wc);
		char* text() const;
		int size() const;
		void split(const char* c, int n, int* data);
		void split(const char* c, int n, double* data);
		std::string toWString();
		const std::string toWString() const;
		//static void get_split_data(xString* ls,  int* data);

	//	friend ostream& operator<<(ostream& os, const xstring& xs);
	private:
		int _n_split(const char* c);
		int len;
		char *wc;
	};
}

// ostream& operator<<(ostream& os, const xdyn::xstring& xs)
// {
// 	os << xs.text();
// 	return os;
// }

// class XDYNAMICS_API xStringList
// {
// 	struct Node
// 	{
// 		xString data;
// 		Node *parent;
// 		Node *child;
// 	};
// public:
// 	xStringList();
// 	xStringList(const xStringList& ls);
// 	~xStringList();
// 
// 	xString& operator++ ();
// 	//void operator= (xStringList& ls);
// 	//xString at(int idx) const;
// 	unsigned int size() const;
// 	void push_back(xString xs);
// 	xString& begin();
// 	bool IsEnd();
// 	Node* Head() const;
// 
// private:
// 	unsigned int sz;
// 	//xString* xstr_list;
// 	Node* curr;
// 	Node* head;
// 	Node* tail;
// 	Node* temp;
// };

#endif