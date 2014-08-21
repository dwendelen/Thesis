#include "unittest.hpp"

namespace cl_cpd
{
	ContextQueue* cqq = NULL;

	template<typename t>
	bool compareSum(t* a, int n, double exp, double delta)
	{
		double sum = 0;
		for(int i = 0; i < n; i++)
			sum += a[i];

		if ((sum < exp + delta) && (sum > exp - delta))
		{
			return true;
		}
		else
		{
			std::cout << exp << " echt: " << sum;
			return false;
		}
	}

	void convertT(T<double> a, T<float>& b)
	{
		b.I = a.I;
		int l = 1;
		for (uint i = 0; i<a.I.size(); i++)
			l *= a.I[i];

		b.Ts = new float[l];
		for (int i = 0; i<l; i++)
			b.Ts[i] = (float)a.Ts[i];
	}

	void convertU(U<double> a, U<float>& b)
	{
		b.I = a.I;
		b.rank = a.rank;
		b.Us = std::vector<float*>(3);
		for(uint i = 0; i<a.Us.size(); i++)
		{
			b.Us[i] = new float[a.rank * a.I[i]];
			for(uint j=0; j< a.rank * a.I[i]; j++)
			{
				b.Us[i][j] = (float)a.Us[i][j];
			}
		}
	}

	template<typename type>
	void testF(Kernel<type>* kernel, BufferFactory<type>* bf, double exp, double delta, bool& bb)
	{
		kernel->compile();
		kernel->setBuffers(bf);
		kernel->run();

		Sum<type> sum;
		sum.nbElements = bf->getNbElementsInSum();
		sum.sum = new type[sum.nbElements];

		bf->readSum(sum);

		bool b = compareSum(sum.sum, sum.nbElements, exp, delta);

		std::cout << kernel->getName() << " ";
		if(b)
			std::cout << "OK";
		else
		{
			bb = false;
			std::cout << "FAIL";
		}

		std::cout << "\n";

		delete sum.sum;
	}

	template<typename type>
	void testFM(AbstractTMapper<type>* m, AbstractFKernel<type>* kernel, AbstractMappedBufferFactory<type>* bf, double exp, double delta, bool& bb)
	{
		m->compile();
		m->setMBuffers(bf);
		m->setTMapped(bf->getTMapped());
		m->setTUnMapped(bf->getTUnMapped());
		m->run();

		kernel->compile();
		kernel->setBuffers(bf);
		kernel->setT(bf->getTMapped());
		kernel->run();

		Sum<type> sum;
		sum.nbElements = bf->getNbElementsInSum();
		sum.sum = new type[sum.nbElements];

		bf->readSum(sum);

		bool b = compareSum(sum.sum, sum.nbElements, exp, delta);

		std::cout << kernel->getName() << " ";
		if(b)
			std::cout << "OK";
		else
		{
			bb = false;
			std::cout << "FAIL";
		}

		std::cout << "\n";

		delete sum.sum;
	}

	template<typename type>
	void testF1D(OneDRangeKernel<type>* kernel, OneDRangeBufferFactory<type>* bf, double exp, double delta, bool& bb)
	{
		kernel->compile();
		kernel->setBuffers(bf);
		kernel->setL(bf->getL());
		kernel->run();

		Sum<type> sum;
		sum.nbElements = bf->getNbElementsInSum();
		sum.sum = new type[sum.nbElements];

		bf->readSum(sum);

		bool b = compareSum(sum.sum, sum.nbElements, exp, delta);

		std::cout << kernel->getName() << " ";
		if(b)
			std::cout << "OK";
		else
		{
			bb = false;
			std::cout << "FAIL";
		}

		std::cout << "\n";

		delete sum.sum;
	}

	void testDouble(T<double> t, U<double> u, double f, double delta, bool& bb)
	{
		Double16x16x16BufferFactory* b = new Double16x16x16BufferFactory(cqq);
		b->init(t, u);

		testF(new Double16x16x16UnMapped(cqq), b, f, delta, bb);

		delete b;

	}

	void testFloat(T<float> t, U<float> u, double f, double delta, bool& bb)
	{
		AbstractBufferFactory<float> * b = new AbstractBufferFactory<float>(cqq, 4);
		AbstractBufferFactory<float> * b8 = new AbstractBufferFactory<float>(cqq, 2);
		AbstractBufferFactory<float> * b4 = new AbstractBufferFactory<float>(cqq, 1);

		/*AbstractMappedBufferFactory<float>* b8M
			= new AbstractMappedBufferFactory<float>(cqq, 2);
		AbstractMappedBufferFactory<float>* b16M
			= new AbstractMappedBufferFactory<float>(cqq, 4);*/

		OneDRangeBufferFactory<float>* b1d16 = new OneDRangeBufferFactory<float>(cqq, 16);
		OneDRangeBufferFactory<float>* b64 = new OneDRangeBufferFactory<float>(cqq, 64);

		b->init(t, u);
		b8->init(t, u);
		b4->init(t, u);

		b1d16->init(t, u);
		b64->init(t, u);

		/*b8M->init(t, u);
		b16M->init(t, u);*/

		testF(new AbstractFKernel<float>(cqq, "float16x16x16", 4), b, f, delta, bb);
		testF(new AbstractFKernel<float>(cqq, "float8x8x8", 2), b8, f, delta, bb);
		testF(new AbstractFKernel<float>(cqq, "float4x4x4", 1), b4, f, delta, bb);


		testF1D(new OneDRangeKernel<float>(cqq, "float", 16), b1d16, f, delta, bb);
		testF(new OneDRangeKernel<float>(cqq, "float64", 64), b64, f, delta, bb);

		/*testFM(	new AbstractTMapper<float>(cqq, "float8x8x8Mapper", 2),
				new AbstractFKernel<float>(cqq, "float8x8x8R", 2),
				b8M, f, delta, bb);

		testFM(	new AbstractTMapper<float>(cqq, "float16x16x16Mapper", 4),
				new AbstractFKernel<float>(cqq, "float16x16x16R", 4),
				b16M, f, delta, bb);

		testFM(	new AbstractTMapper<float>(cqq, "float16x16x16MapperI", 4),
				new AbstractFKernel<float>(cqq, "float16x16x16RI", 4),
				b16M, f, delta, bb);*/

		delete b;
		delete b8;
		delete b4;

		delete b1d16;
		delete b64;

		/*delete b8M;
		delete b16M;*/

	}

	bool compareG(U<double> r, U<double> e, double delta)
	{
		bool kill = true;
		for(uint i = 0; i < 3; i++)
		{
			for(uint j = 0; j < (r.rank * r.I[i]); j++)
			{
				if ((r.Us[i][j] < e.Us[i][j] + delta) && (r.Us[i][j] > e.Us[i][j] - delta))
				{
					//niks doen
				}
				else
				{
					std::cout << "In G " << (i+1);
					std::cout << "El " << j;
					std::cout << e.Us[i][j] << " echt: " << r.Us[i][j];
					kill = false;
					break;
				}
			}
		}
		return kill;
	}

	void testG(T<double> t, U<double> u, double f, double deltaF, U<double> g, double deltaG, bool& bb)
	{
		AbstractFGBufferFactory<double> *b = new AbstractFGBufferFactory<double>(cqq, 4);
		b->init(t, u);

		AbstractFGKernel<double>* k1 = new AbstractFGKernel<double>(cqq, "double16x16x16FG", 4);
		AbstractGKernel<double>* k2 = new AbstractGKernel<double>(cqq, "double16x16x16G");

		


        k1->compile();
		k1->setBuffers(b);
		std::cout << "Buff GF set";
		k1->run();
		
		std::cout << "GF ran";

		Sum<double> sum;
		sum.nbElements = b->getNbElementsInSum();
		sum.sum = new double[sum.nbElements];

		b->readSum(sum);

		bool bbb = compareSum(sum.sum, sum.nbElements, f, deltaF);

		std::cout << k1->getName() << " ";
		if(bbb)
			std::cout << "OK";
		else
		{
			bb = false;
			std::cout << "FAIL";
		}

		std::cout << "\n";

		delete sum.sum;

        k2->compile();
		k2->setBuffers(b);
		std::cout << "Buff G set";
		k2->run();

		U<double> uu;
		uu.I = u.I;
		uu.rank = u.rank;

		uu.Us = std::vector<double*>(3);
		uu.Us[0] = new double[uu.I[0] * uu.rank];
		uu.Us[1] = new double[uu.I[1] * uu.rank];
		uu.Us[2] = new double[uu.I[2] * uu.rank];

		b->readG(uu);

		bbb = compareG(uu, g, deltaG);

		std::cout << k2->getName() << " ";
		if(bbb)
			std::cout << "OK";
		else
		{
			bb = false;
			std::cout << "FAIL";
		}

		std::cout << "\n";

		delete uu.Us[0];
		delete uu.Us[1];
		delete uu.Us[2];

		delete b;
		delete k1;
		delete k2;
	}

	bool UnitTest::test(T<double> t, U<double> u, double f, double deltaF, U<double> g, double deltaG)
	{
		cqq = new ContextQueue();
		cqq->init(false);

		std::cout << "\n";
		std::cout << "\n";
		std::cout << "\n";

		bool bb = true;
		testDouble(t, u, f, deltaF, bb);

		T<float> tt;
		U<float> uu;

		convertT(t, tt);
		convertU(u, uu);

		testFloat(tt, uu, f, deltaF, bb);

		testG(t, u, f, deltaF, g, deltaG, bb);

		delete cqq;

		return bb;
	}

}


