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
		for(int i = 0; i<a.Us.size(); i++)
		{
			b.Us[i] = new float[a.rank * a.I[i]];
			for(uint j=0; j< a.rank * a.I[i]; j++)
			{
				b.Us[i][j] = (float)a.Us[i][j];
			}
		}
	}

	template<typename type>
	void testF(AbstractFKernel<type>* kernel, AbstractBufferFactory<type>* bf, double exp, double delta, bool& bb)
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
		b->init(t, u);
		b8->init(t, u);
		b4->init(t, u);

		testF(new AbstractFKernel<float>(cqq, "float16x16x16", 4), b, f, delta, bb);
		testF(new AbstractFKernel<float>(cqq, "float8x8x8", 2), b8, f, delta, bb);
		testF(new AbstractFKernel<float>(cqq, "float4x4x4", 1), b4, f, delta, bb);

		delete b;
		delete b8;
		delete b4;

	}

	bool UnitTest::test(T<double> t, U<double> u, double f, double delta)
	{
		cqq = new ContextQueue();
		cqq->init(false);

		std::cout << "\n";
		std::cout << "\n";
		std::cout << "\n";

		bool bb = true;
		testDouble(t, u, f, delta, bb);

		T<float> tt;
		U<float> uu;

		convertT(t, tt);
		convertU(u, uu);

		testFloat(tt, uu, f, delta, bb);

		delete cqq;

		return bb;
	}

}


