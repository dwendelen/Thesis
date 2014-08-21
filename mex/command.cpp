/*
 * command.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */


#include "command.hpp"

namespace cl_cpd
{

	ContextQueue *cq = NULL;
	Double16x16x16UnMapped *f = NULL;
	Double16x16x16BufferFactory *b = NULL;

	std::vector<mxArray*> TestCommand::handle()
	{
		UnitTest ut;
		bool b = ut.test(t.val, u.val, f.val, deltaF.val, g.val, deltaG.val);

		std::vector<mxArray*> r(1);
		r[0] = mxCreateLogicalScalar(b);
		return r;
	}

	std::vector<mxArray*> InitCommand::handle()
	{
		delete cq;
		delete f;

        cq = new ContextQueue();
		cq->init(profile.val);

        f = new Double16x16x16UnMapped(cq);
		f->compile();

        delete b;
        b = new Double16x16x16BufferFactory(cq);

		return std::vector<mxArray*>(0);
	}

	std::vector<mxArray*> SetTCommand::handle()
	{
		if(f == NULL || b == NULL)
			mexErrMsgTxt("cl_cpd is not initialised.");

		b->init(t.val, u.val);

		f->setT(b->getT());
		f->setRank(b->getRank());
		f->setU(b->getU());
		f->setI(b->getI());
		f->setSum(b->getSum());


		return std::vector<mxArray*>(0);
	}

	std::vector<mxArray*> SetUCommand::handle()
	{
		std::cout << "    mroeu    " ;
		if(f == NULL || b == NULL)
			mexErrMsgTxt("cl_cpd is not initialised.");

		b->updateU(u.val);

		return std::vector<mxArray*>(0);
	}

	std::vector<mxArray*> RunCommand::handle()
	{
		if(f == NULL || b == NULL)
			mexErrMsgTxt("cl_cpd is not initialised.");

		f->run();
		std::cout << "\n\n" << b->getNbElementsInSum() << "\n\n";
		mxArray* m = mxCreateDoubleMatrix(b->getNbElementsInSum(), 1, mxREAL);

		Sum<double> s;
		s.nbElements = mxGetNumberOfElements(m);
		s.sum = mxGetPr(m);

		b->readSum(s);

		std::vector<mxArray*> r = std::vector<mxArray*>(1);
		r[0] = m;
		return r;
	}

	std::vector<mxArray*> TimeCommand::handle()
	{
		std::vector<mxArray*> r (1);
		r[0] = mxCreateDoubleScalar((double)f->getExecutionTimeLastRun());
		return r;
	}

	std::vector<mxArray*> InvlRCommand::handle()
	{
		std::vector<graph> gs;
		invlR(gs);

		GraphConverter gc;

		std::vector<mxArray*> r (1);
		r[0] = gc.convert(gs);
		return r;
	}

	std::vector<mxArray*> MeasureFCommand::handle()
	{
		Data data;
		measureF(data);

		DataConverter dc;

		std::vector<mxArray*> r (1);
		r[0] = dc.convert(data);
		return r;
	}

	bool BoolParameter::validate(const mxArray* input)
	{
		return mxIsLogicalScalar(input);
	}
	bool BoolParameter::convert(const mxArray* input)
	{
		return mxGetLogicals(input)[0];
	}

	bool DoubleParameter::validate(const mxArray* input)
	{
		if(!mxIsDouble(input))
			return false;

		return mxGetNumberOfElements(input) == 1;
	}
	double DoubleParameter::convert(const mxArray* input)
	{
		return mxGetPr(input)[0];
	}

	bool StringParameter::validate(const mxArray* input)
	{
		if(!mxIsChar(input))
			return false;

		if(mxGetM(input) != 1)
			return false;

		return true;
	}
	std::string StringParameter::convert(const mxArray* input)
	{
		return std::string(mxArrayToString(input));
	}

	bool TParameter::validate(const mxArray* input)
	{
		return mxIsDouble(input);
	}
	T<double> TParameter::convert(const mxArray* input)
	{
		T<double> t;
		mwSize dims = mxGetNumberOfDimensions(input);
		t.I = std::vector<size_t>(dims);
		for(size_t i = 0; i < dims; i++)
		{
			t.I[i] = mxGetDimensions(input)[i];
		}

		t.Ts = mxGetPr(input);

		return t;
	}

	bool UParameter::validate(const mxArray* input)
	{
		if(!mxIsCell(input))
			return false;

		mwSize nbElements = mxGetNumberOfElements(input);

		if(nbElements < 1)
			return false;

		if(mxIsEmpty(mxGetCell(input, 0)))
			return false;

		if(!mxIsDouble(mxGetCell(input, 0)))
			return false;

		size_t R = mxGetN(mxGetCell(input, 0));

		for(size_t i = 1; i < nbElements; i++)
		{
			if(mxIsEmpty(mxGetCell(input, i)))
				return false;

			if(!mxIsDouble(mxGetCell(input, i)))
				return false;

			if(mxGetN(mxGetCell(input, i)) != R)
				return false;
		}

		return mxIsDouble(mxGetCell(input, 0));
	}
	U<double> UParameter::convert(const mxArray* input)
	{
		U<double> u;
		mwSize dims = mxGetNumberOfElements(input);
		u.I = std::vector<size_t>(dims);
		u.Us = std::vector<double*>(dims);

		for(size_t i = 0; i < dims; i++)
		{
			u.I[i] = mxGetM(mxGetCell(input, i));
			u.Us[i] = mxGetPr(mxGetCell(input, i));
		}
		u.rank = mxGetN(mxGetCell(input, 0));

		return u;
	}

	void convertLine(line l, mxArray* lines, int idxL)
	{
		mxArray* name = mxCreateString(l.name.c_str());
		mxArray* x = mxCreateDoubleMatrix(l.x.size(), 1, mxREAL);
		mxArray* y = mxCreateDoubleMatrix(l.y.size(), 1, mxREAL);

		mxSetField(lines, idxL, "name", name);
		mxSetField(lines, idxL, "x", x);
		mxSetField(lines, idxL, "y", y);

		std::copy(l.x.begin(), l.x.end(), mxGetPr(x));
		std::copy(l.y.begin(), l.y.end(), mxGetPr(y));
	}

	void convertGraph(graph g, mxArray* graphs, int idxG)
	{
		mxArray* val = mxCreateDoubleScalar(g.val);

		const char* f[] = {"name", "x", "y"};

		mxArray* lines =
			mxCreateStructMatrix(g.lines.size(),1, 3, f);

		mxSetField(graphs, idxG, "val", val);
		mxSetField(graphs, idxG, "lines", lines);

		for(size_t idxL = 0; idxL < g.lines.size(); idxL++)
		{
			convertLine(g.lines[idxL], lines, idxL);
		}

	}

	mxArray* GraphConverter::convert(const std::vector<graph> input)
	{
		const char* f[] =  {"val", "lines"};

		mxArray* graphs = mxCreateStructMatrix(input.size(), 1, 2, f);

		for(size_t idxG = 0; idxG < input.size(); idxG++)
		{
			convertGraph(input[idxG], graphs, idxG);
		}

		return graphs;
	}


	mxArray* DataConverter::convert(Data data)
	{
		size_t s[] = {data.I.size, data.R.size(), data.nbKernels};
		mxArray* r = mxCreateNumericArray(3, s, mxDOUBLE_CLASS, mxREAL);

		std::copy(data.data, data.data + data.size(), mxGetPr(r));

		return r;
	}



	/*Sum SumConverter::convert(const mxArray* input)
	{
		Sum r;
		r.nbElements = mxGetNumberOfElements(input);
		r.sum =
		mxArray* m = mxCreateDoubleMatrix(input->nbElements, 1, mxREAL);
		mxSetPr(m, input->sum);
		return m;
	}*/

}

