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
#include "double8x8x8I.hpp"

using namespace cl_cpd;
using namespace std;

double bigT[320*320*320];
double bigU[320*10000];

float bigTf[320*320*320];
float bigUf[320*10000];

/*
Double16x16x16UnMapped* f = NULL;
Double16x16x16ReMapped* r = NULL;
Double16x16x16Isolated* i = NULL;
AbstractFKernel<double>* p = NULL;

Double16x16x16BufferFactory* b = NULL;
Double16x16x16FGBufferFactory* bg = NULL;

Double8x8x8ReMapped* r8 = NULL;
Double8x8x8Isolated* i8 = NULL;
Double8x8x8BufferFactory* b8 = NULL;

Double16x16x16G* g = NULL;
*/
//
//OneDRangeBufferFactory<float>* b1d1 = NULL;
OneDRangeBufferFactory<float>* b1d32 = NULL;
OneDRangeBufferFactory<float>* b1d64 = NULL;

//OneDRangeKernel<float>* f1d1 = NULL;
OneDRangeKernel<float>* f1d32 = NULL;
OneDRangeKernel<float>* f1d64 = NULL;

OneDRangeKernel<float>* f64 = NULL;

AbstractBufferFactory<float>* b4 = NULL;
AbstractBufferFactory<float>* b8 = NULL;
AbstractBufferFactory<float>* b16 = NULL;

AbstractFKernel<float>* f4 = NULL;
AbstractFKernel<float>* f8 = NULL;
AbstractFKernel<float>* f16 = NULL;

template<typename T>
void run(Kernel<T>* kernel, string name, double ops){
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


	/*T<double> t;
	t.Ts = bigT;
	t.I = vector<size_t>();
	t.I.push_back(I16);
	t.I.push_back(I16);
	t.I.push_back(I16);

	U<double> u;
	u.Us = vector<double*>();
	u.Us.push_back(bigU);
	u.Us.push_back(bigU);
	u.Us.push_back(bigU);
	u.I = t.I;
	u.rank = R;*/

	T<float> t;
	t.Ts = bigTf;
	t.I = vector<size_t>();
	t.I.push_back(I16);
	t.I.push_back(I16);
	t.I.push_back(I16);

	U<float> u;
	u.Us = vector<float*>();
	u.Us.push_back(bigUf);
	u.Us.push_back(bigUf);
	u.Us.push_back(bigUf);
	u.I = t.I;
	u.rank = R;

	/*b1d1->init(t, u);
	f1d1->setBuffers(b1d1);
	f1d1->setL(b1d1->getL());
	run(f1d1, "Float1", ops);*/

	b1d32->init(t, u);
	f1d32->setBuffers(b1d32);
	f1d32->setL(b1d32->getL());
	run(f1d32, "Float 32", ops);

	b1d64->init(t, u);
	f1d64->setBuffers(b1d64);
	f1d64->setL(b1d64->getL());
	run(f1d64, "Float 64", ops);

	f64->setBuffers(b1d64);
	run(f64, "Float64", ops);

	b4->init(t, u);
	f4->setBuffers(b4);
	run(f4, "Float4x4x4", ops);

	b8->init(t, u);
	f8->setBuffers(b8);
	run(f8, "Float8x8x8", ops);

	b16->init(t, u);
	f16->setBuffers(b16);
	run(f16, "Float16x16x16", ops);
/*
	b->init(t, u);
	f->setBuffers(b);
	r->setBuffers(b);
	i->setBuffers(b);
	p->setBuffers(b);

	run(f, "16x16x16 Unmapped", ops);
	run(r, "16x16x16 Pointer", ops);
	run(r, "16x16x16 Remapped", ops);
	run(i, "16x16x16 Isolated", ops);

	int R16 = R;
	if(R % 16 != 0)
		R16 = (R/16)*16 + 16;

	U<double> u2 = u;
	u2.rank = R16;

	bg->init(t, u2);
	g->setBuffers(bg);
	run(g, "16x16x16 Gradient", ops);

	int I8 = I;
	if(I % 8 != 0)
		I8 = (I/8)*8 + 8;

	t.I = vector<size_t>();
	t.I.push_back(I8);
	t.I.push_back(I8);
	t.I.push_back(I8);
	u.I = t.I;

	b8->init(t, u);
	r8->setBuffers(b8);
	i8->setBuffers(b8);

	run(r8, "8x8x8 Remapped", ops);
	//run(i8, "8x8x8 Isolated", ops); I needs extra memory
	 */
}

void dooo()
{
	ContextQueue* cq = new ContextQueue();
	cq->init(true);

	//b1d1 = new OneDRangeBufferFactory<float>(cq, 1);
	b1d32 = new OneDRangeBufferFactory<float>(cq, 32);
	b1d64 = new OneDRangeBufferFactory<float>(cq, 64);

	/*f1d1 = new OneDRangeKernel<float>(cq, "float", 1);
	f1d1->compile();*/
	f1d32 = new OneDRangeKernel<float>(cq, "float", 32);
	f1d32->compile();
	f1d64 = new OneDRangeKernel<float>(cq, "float", 64);
	f1d64->compile();

	f64 = new OneDRangeKernel<float>(cq, "float64", 64);
	f64->compile();

	b4 = new AbstractBufferFactory<float>(cq, 1);
	b8 = new AbstractBufferFactory<float>(cq, 2);
	b16 = new AbstractBufferFactory<float>(cq, 4);

	f4 = new AbstractFKernel<float>(cq, "float4x4x4", 1);
	f4->compile();
	f8 = new AbstractFKernel<float>(cq, "float8x8x8", 2);
	f8->compile();
	f16 = new AbstractFKernel<float>(cq, "float16x16x16", 4);
	f16->compile();

	doo(16, 16);
	doo(6000, 16);
	doo(16, 320);
	doo(6000, 320);

	//delete b1d1;
	delete b1d32;
	delete b1d64;

	//delete f1d1;
	delete f1d32;
	delete f1d64;

	delete f64;

	delete b4;
	delete b8;
	delete b16;

	delete f4;
	delete f8;
	delete f16;


	/*
	b = new Double16x16x16BufferFactory(cq);
	b8 = new Double8x8x8BufferFactory(cq);
	bg = new Double16x16x16FGBufferFactory(cq);

	f = new Double16x16x16UnMapped(cq);
	f->compile();
	r = new Double16x16x16ReMapped(cq);
	r->compile();
	i = new Double16x16x16Isolated(cq);
	i->compile();
	r8 = new Double8x8x8ReMapped(cq);
	r8->compile();
	i8 = new Double8x8x8Isolated(cq);
	i8->compile();

	p = new AbstractFKernel<double>(cq, "double16x16x16P", 4);
	p->compile();

	g = new Double16x16x16G(cq);
	g->compile();

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
	delete i8;
	*/
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

