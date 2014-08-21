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
		Is.push_back(16);
		Is.push_back(128);
		Is.push_back(320);

		vector<int> Rs;
		for(int i = 1; i <= 160; i++)
			Rs.push_back(i);

		g.val = 0;
		g.lines = vector<line>(3);
		g.lines[0].name = "I = 16";
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
}
