// #ifndef XSTRING_H
// #define XSTRING_H
// 
// #include "xdynamics_decl.h"
// //#include "xStringList.h"
// 
// //class xStringList;
// 
// class XDYNAMICS_API xString
// {
// public:
// 	xString();
// 	xString(const wchar_t* _wc);
// 	xString(const xString& _xs);
// 	~xString();
// 
// 	void operator= (const xString& _xs);
// 	bool operator== (const wchar_t* _wc);
// 	xString operator+ (const xString& _xs);
// 	xString operator+ (const wchar_t* _wc);
// 	wchar_t* text() const;
// 	int size() const;
// 	void split(const wchar_t* c, int n, int* data);
// 	void split(const wchar_t* c, int n, double* data);
// 	std::wstring toWString();
// 	const std::wstring toWString() const;
// 	//static void get_split_data(xString* ls,  int* data);
// 
// private:
// 	int _n_split(const wchar_t* c);
// 	int len;
// 	wchar_t *wc;
// };
// 
// // class XDYNAMICS_API xStringList
// // {
// // 	struct Node
// // 	{
// // 		xString data;
// // 		Node *parent;
// // 		Node *child;
// // 	};
// // public:
// // 	xStringList();
// // 	xStringList(const xStringList& ls);
// // 	~xStringList();
// // 
// // 	xString& operator++ ();
// // 	//void operator= (xStringList& ls);
// // 	//xString at(int idx) const;
// // 	unsigned int size() const;
// // 	void push_back(xString xs);
// // 	xString& begin();
// // 	bool IsEnd();
// // 	Node* Head() const;
// // 
// // private:
// // 	unsigned int sz;
// // 	//xString* xstr_list;
// // 	Node* curr;
// // 	Node* head;
// // 	Node* tail;
// // 	Node* temp;
// // };
// 
// #endif