/*
 * meas.cpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#include <iostream>

#include "double16x16x16.hpp"
#include "double16x16x16R.hpp"

using namespace cl_cpd;
using namespace std;

double bigT[320*320*320];
double bigU[320*10000];

Double16x16x16UnMapped* f = NULL;
Double16x16x16ReMapped* r = NULL;
Double16x16x16BufferFactory* b = NULL;

void run(Kernel* kernel, string name, double ops){
	cout << name << "\n";

	kernel->run();
	kernel->run();

	cout << (kernel->getExecutionTimeLastRun() / 1000000) << "\n";
	cout << (ops/kernel->getExecutionTimeLastRun()) << "\n";
}

void doo(int R, int I)
{
    double ops = ((double)R + 2) * (double)I*(double)I*(double)I;
	cout << "\nR: " << R << " I: " << I << "\n";

	if(I % 16 != 0)
		I = (I/16)*16 + 16;

	T t;
	t.Ts = bigT;
	t.I = vector<size_t>();
	t.I.push_back(I);
	t.I.push_back(I);
	t.I.push_back(I);

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

	run(f, "16x16x16 Unmapped", ops);
	run(r, "16x16x16 Remapped", ops);
}

void dooo()
{
	ContextQueue* cq = new ContextQueue();
	cq->init(true);

	b = new Double16x16x16BufferFactory(cq);

	f = new Double16x16x16UnMapped(cq);
	f->compile();
	r = new Double16x16x16ReMapped(cq);
	r->compile();

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

	delete b;
	delete f;
}

