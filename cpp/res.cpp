#include "res.hpp"

double bT[320*320*320];
double bU[320*10000];

float bTf[320*320*320];
float bUf[320*10000];

using namespace std;

namespace cl_cpd
{
	void invlR(std::vector<graph>& gs)
	{
		graph g;

		ContextQueue cq;
		cq.init(true);
		AbstractBufferFactory<float> b(&cq, 4);
		AbstractFKernel<float> f(&cq, "float16x16x16", 4);
		f.compile();

		vector<int> Is;
		Is.push_back(64);
		Is.push_back(128);
		Is.push_back(320);

		vector<int> Rs;
		for(int i = 1; i <= 4000 ; i += 1)
			Rs.push_back(i);

		g.val = 0;
		g.lines = vector<line>(3);
		g.lines[0].name = "I = 64";
		g.lines[1].name = "I = 128";
		g.lines[2].name = "I = 320";

		T<float> t;
		t.Ts = bTf;

		U<float> u;
		u.Us = vector<float*>();
		u.Us.push_back(bUf);
		u.Us.push_back(bUf);
		u.Us.push_back(bUf);

		for(size_t i = 0; i < Is.size(); i++)
		{
			int I = Is[i];

			t.I = vector<size_t>();
			t.I.push_back(I);
			t.I.push_back(I);
			t.I.push_back(I);

			u.I = t.I;


			for(size_t r = 0; r < Rs.size(); r++)
			{
				int R = Rs[r];
				u.rank = R;

				b.init(t, u);
				f.setBuffers(&b);

				f.run();
				f.run();

				g.lines[i].x.push_back(R);
				g.lines[i].y.push_back(f.getExecutionTimeLastRun());
			}
		}

		gs.push_back(g);
	}

	void measureF(Data& data)
	{
		ContextQueue cq;
		cq.init(true);

		AbstractBufferFactory<float> b(&cq, 4);
		AbstractBufferFactory<float> b8(&cq, 2);

		AbstractFKernel<float> f(&cq, "float16x16x16", 4);
		AbstractFKernel<float> fr(&cq, "float16x16x16R", 4);
		AbstractFKernel<float> fi(&cq, "float16x16x16I", 4);

		AbstractFKernel<float> f8(&cq, "float8x8x8", 2);
		AbstractFKernel<float> f8r(&cq, "float8x8x8R", 2);

		f.compile();
		fi.compile();
		fr.compile();

		f8.compile();
		f8r.compile();

		data.R = vector<int>();
		data.R.push_back(8);
		data.R.push_back(16);
		data.R.push_back(400);
		data.R.push_back(4000);

		data.I = vector<int>(40);
		vector<int> I8(40);
		vector<int> I16(40);

		for(int i = 8; i <= 320 ;)
		{
			I8.push_back(i);
			data.I.push_back(i);
			i += 8;
			I16.push_back(i);
			I16.push_back(i);
			I8.push_back(i);
			data.I.push_back(i);
			i += 8;
		}

		T<float> t16;
		t16.Ts = bTf;

		U<float> u16;
		u16.Us = vector<float*>();
		u16.Us.push_back(bUf);
		u16.Us.push_back(bUf);
		u16.Us.push_back(bUf);

		T<float> t8 = t16;
		U<float> u8 = u16;

		data.nbKernels = 5;
		cout << data.size();
		data.data = new double[data.size()];
		double* p = data.data;

		for(size_t i = 0; i < data.I.size(); i++)
		{
			cout << "\nI: " << i << "\n";

			t16.I = vector<size_t>();
			t16.I.push_back(I16[i]);
			t16.I.push_back(I16[i]);
			t16.I.push_back(I16[i]);

			u16.I = t16.I;

			t8.I = vector<size_t>();
			t8.I.push_back(I8[i]);
			t8.I.push_back(I8[i]);
			t8.I.push_back(I8[i]);

			u8.I = t8.I;

			cout << "voor R";
			for(size_t r = 0; r < data.R.size(); r++)
			{
				int R = data.R[r];
				u16.rank = R;
				u8.rank = R;

				b.init(t16, u16);
				f.run();
				f.run();
				*p = f.getExecutionTimeLastRun();
				p++;

				fr.setBuffers(&b);
				fr.run();
				fr.run();
				*p = f.getExecutionTimeLastRun();
				p++;

				fi.setBuffers(&b);
				fi.run();
				fi.run();
				*p = f.getExecutionTimeLastRun();
				p++;

				b8.init(t8,u8);
				f8.setBuffers(&b8);
				f8.run();
				f8.run();
				*p = f.getExecutionTimeLastRun();
				p++;

				f8r.setBuffers(&b8);
				f8r.run();
				f8r.run();
				*p = f.getExecutionTimeLastRun();
				p++;
			}
		}
	}
}
