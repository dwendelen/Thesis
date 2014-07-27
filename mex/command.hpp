/*
 * command.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef CL_CPD_COMMAND_HPP_
#define CL_CPD_COMMAND_HPP_

#include <iostream>

#include "mex.h"
#include "common.hpp"

namespace cl_cpd
{
	class Command
	{
	public:
		virtual std::string getString() = 0;
		virtual void handle(int nlhs, mxArray *plhs[],
				int nrhs, const mxArray *prhs[]) = 0;
		virtual ~Command() {}
	};

	template <typename TypeToConvertTo>
	class Converter
	{
	public:
		virtual bool validate(const mxArray* input) = 0;
		virtual TypeToConvertTo convert(const mxArray* input) = 0;
	};

	class CStringConverter: public Converter<char*>
	{
	public:
		bool validate(const mxArray* input)
		{
			if(!mxIsChar(input))
				return false;

			if(mxGetM(input) != 1)
				return false;

			return true;
		}
		char* convert(const mxArray* input)
		{
			return mxArrayToString(input);
		}
	};

	class TConverter: public Converter<T*>
	{
	public:

		bool validate(const mxArray* input)
		{
			return mxIsDouble(input);
		}
		T* convert(const mxArray* input)
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
	};

	class UConverter: public Converter<U*>
	{
	public:

		bool validate(const mxArray* input)
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
		U* convert(const mxArray* input)
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
	};
}

#endif /* CL_CPD_COMMAND_HPP_ */
