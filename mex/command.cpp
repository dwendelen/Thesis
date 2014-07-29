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
		f->setR(b->getR());
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

		Sum s;
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

	bool BoolParameter::validate(const mxArray* input)
	{
		return mxIsLogicalScalar(input);
	}
	bool BoolParameter::convert(const mxArray* input)
	{
		return mxGetLogicals(input)[0];
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
	T TParameter::convert(const mxArray* input)
	{
		T t;
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
	U UParameter::convert(const mxArray* input)
	{
		U u;
		mwSize dims = mxGetNumberOfElements(input);
		u.I = std::vector<size_t>(dims);
		u.Us = std::vector<double*>(dims);

		for(size_t i = 0; i < dims; i++)
		{
			u.I[i] = mxGetM(mxGetCell(input, i));
			u.Us[i] = mxGetPr(mxGetCell(input, i));
		}
		u.R = mxGetN(mxGetCell(input, 0));

		return u;
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

