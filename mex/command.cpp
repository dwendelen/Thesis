/*
 * command.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */


#include "command.hpp"

namespace cl_cpd
{
	ContextQueue cq;
	Double16x16x16UnMapped f(&cq);
	Double16x16x16BufferFactory b(&cq);

	std::vector<mxArray*> InitCommand::handle(std::vector<const mxArray*> input)
	{
		std::cout << "enter init";
		bool profile = BoolConverter().convert(input[0]);
        
        cq = ContextQueue();
		cq.init(profile);
        
        f = Double16x16x16UnMapped(&cq);
		f.compile();

		std::cout << "exit init";
		return std::vector<mxArray*>(0);
	}

	std::vector<mxArray*> SetTCommand::handle(std::vector<const mxArray*> input)
	{
		T* t = TConverter().convert(input[0]);
		U* u = UConverter().convert(input[1]);
        
        b = Double16x16x16BufferFactory(&cq);
		b.init(*t, *u);

		f.setT(b.getT());
		f.setR(b.getR());
		f.setU(b.getU());
		f.setI(b.getI());
		f.setSum(b.getSum());

		delete t;
		delete u;

		return std::vector<mxArray*>(0);
	}

	std::vector<mxArray*> RunCommand::handle(std::vector<const mxArray*> input)
	{
		f.run();

		b.readSum();
		mxArray* s = SumConverter().convert(b.getSumArray());

		std::vector<mxArray*> r = std::vector<mxArray*>(1);
		r[0] = s;
		return r;
	}

	bool BoolConverter::validate(const mxArray* input)
	{
		return mxIsLogicalScalar(input);
	}
	bool BoolConverter::convert(const mxArray* input)
	{
		return mxGetLogicals(input)[0];
	}

	bool CStringConverter::validate(const mxArray* input)
	{
		if(!mxIsChar(input))
			return false;

		if(mxGetM(input) != 1)
			return false;

		return true;
	}
	char* CStringConverter::convert(const mxArray* input)
	{
		return mxArrayToString(input);
	}

	bool TConverter::validate(const mxArray* input)
	{
		return mxIsDouble(input);
	}
	T* TConverter::convert(const mxArray* input)
	{
		T* t = new T;
		mwSize dims = mxGetNumberOfDimensions(input);
		t->I = std::vector<size_t>(dims);
		for(size_t i = 0; i < dims; i++)
		{
			t->I[i] = mxGetDimensions(input)[i];
		}

		t->Ts = mxGetPr(input);

		return t;
	}

	bool UConverter::validate(const mxArray* input)
	{
		if(!mxIsCell(input))
			return false;

		mwSize dims = mxGetNumberOfElements(input);

		if(dims < 1)
			return false;

		size_t R = mxGetN(mxGetCell(input, 0));

		for(size_t i = 1; i < dims; i++)
		{
			if(mxGetN(mxGetCell(input, i)) != R)
				return false;

			if(!mxIsDouble(mxGetCell(input, i)))
				return false;
		}

		return mxIsDouble(mxGetCell(input, 0));
	}
	U* UConverter::convert(const mxArray* input)
	{
		U* u = new U;
		mwSize dims = mxGetNumberOfElements(input);
		u->I = std::vector<size_t>(dims);
		u->Us = std::vector<double*>(dims);
		for(size_t i = 0; i < dims; i++)
		{
			u->I[i] = mxGetM(mxGetCell(input, i));
			u->Us[i] = mxGetPr(mxGetCell(input, i));
		}
		u->R = mxGetN(mxGetCell(input, 0));

		return u;
	}

	mxArray* SumConverter::convert(const Sum* input)
	{
		mxArray* m = mxCreateDoubleMatrix(input->nbElements, 1, mxREAL);
		mxSetPr(m, input->sum);
		return m;
	}

}

