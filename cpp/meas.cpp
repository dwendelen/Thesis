/*
 * meas.cpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#include <iostream>

#include "double16x16x16.hpp"

using namespace cl_cpd;

int main()
{
	try{
	T t;
	t.Ts = new double[32*16*64];
	t.I = std::vector<size_t>();
	t.I.push_back(32);
	t.I.push_back(16);
	t.I.push_back(64);

	U u;
	u.I = std::vector<size_t>();
	u.I.push_back(32);
	u.I.push_back(16);
	u.I.push_back(64);
	u.R = 8;
	u.Us = std::vector<double*>();
	u.Us.push_back(new double[32*8]);
	u.Us.push_back(new double[16*8]);
	u.Us.push_back(new double[64*8]);

	ContextQueue cq;
	cq.init(true);

	Double16x16x16BufferFactory b (&cq);
	b.init(t, u);

	Double16x16x16UnMapped f (&cq);
	f.compile();


	f.setT(b.getT());
	f.setR(b.getR());
	f.setU(b.getU());
	f.setI(b.getI());
	f.setSum(b.getSum());

	f.run();


	}
	catch (cl::Error &e)
	{
		std::cout << "Exception OpenCL: " << e.what() << " code: " << e.err();
	}

	std::cout << "OK";
}


