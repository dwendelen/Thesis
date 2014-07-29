/*
 * meas.cpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#include <iostream>

#include "double16x16x16.hpp"
#include "double16x16x16R.hpp"
#include "double16x16x16I.hpp"
#include "double8x8x8R.hpp"

using namespace cl_cpd;
using namespace std;

double bigT[320*320*320];
double bigU[320*10000];

Double16x16x16UnMapped* f = NULL;
Double16x16x16ReMapped* r = NULL;
Double16x16x16Isolated* i = NULL;
Double16x16x16BufferFactory* b = NULL;

Double8x8x8ReMapped* r8 = NULL;
Double8x8x8BufferFactory* b8 = NULL;

void run(Kernel* kernel, string name, double ops){
	cout << name << "\n";

	kernel->run();
	kernel->run();

	cout << (kernel->getExecutionTimeLastRun() / 1000000) << "\n";
	cout << (ops/kernel->getExecutionTimeLastRun()) << "\n";
}

void doo(int R, int I)
{
    double ops = (3*(double)R + (2*(double)R/(double)I) + 3)
    		* (double)I*(double)I*(double)I;

	cout << "\nR: " << R << " I: " << I << "\n";

	int I16 = I;
	if(I % 16 != 0)
		I16 = (I/16)*16 + 16;

	T t;
	t.Ts = bigT;
	t.I = vector<size_t>();
	t.I.push_back(I16);
	t.I.push_back(I16);
	t.I.push_back(I16);

	U u;
	u.Us = vector<double*>();
	u.Us.push_back(bigU);
	u.Us.push_back(bigU);
	u.Us.push_back(bigU);
	u.I = t.I;
	u.R = R;

	b->init(t, u);
	f->setT(b->getT());
	f->setR(b->getR());
	f->setU(b->getU());
	f->setI(b->getI());
	f->setSum(b->getSum());

	r->setT(b->getT());
	r->setR(b->getR());
	r->setU(b->getU());
	r->setI(b->getI());
	r->setSum(b->getSum());

	i->setT(b->getT());
	i->setR(b->getR());
	i->setU(b->getU());
	i->setI(b->getI());
	i->setSum(b->getSum());

	run(f, "16x16x16 Unmapped", ops);
	run(r, "16x16x16 Remapped", ops);
	run(i, "16x16x16 Isolated", ops);

	int I8 = I;
	if(I % 8 != 0)
		I8 = (I/8)*8 + 8;

	t.I = vector<size_t>();
	t.I.push_back(I8);
	t.I.push_back(I8);
	t.I.push_back(I8);
	u.I = t.I;

	b8->init(t, u);
	r8->setT(b->getT());
	r8->setR(b->getR());
	r8->setU(b->getU());
	r8->setI(b->getI());
	r8->setSum(b->getSum());

	run(r8, "8x8x8 Remapped", ops);
}

void dooo()
{
	ContextQueue* cq = new ContextQueue();
	cq->init(true);

	b = new Double16x16x16BufferFactory(cq);
	b8 = new Double8x8x8BufferFactory(cq);

	f = new Double16x16x16UnMapped(cq);
	f->compile();
	r = new Double16x16x16ReMapped(cq);
	r->compile();
	i = new Double16x16x16Isolated(cq);
	i->compile();
	r8 = new Double8x8x8ReMapped(cq);
	r8->compile();

	doo(4,1);
	doo(6000,1);
	doo(4,10);
	doo(600,10);

	doo(16,100);
	doo(4,100);
	doo(16,320);
	doo(4,320);

	doo(6000,16);
	doo(6000,100);
	doo(6000,320);

	doo(1024,320);
	doo(100,320);
	doo(128,320);
	doo(1,320);

	doo(1,1);
	doo(1,4);
	doo(1,16);
	doo(1,64);
	doo(1,256);
	doo(1,320);

	delete cq;
	delete b;
	delete f;
	delete r;
	delete i;
	delete b8;
	delete r8;
}
int main()
{
	try{
		dooo();
	}
	catch (cl::Error &e)
	{
		cerr << "Exception OpenCL: " << e.what() << " code: " << e.err();
	}
}

