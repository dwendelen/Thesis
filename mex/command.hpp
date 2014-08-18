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
#include "../cpp/common.hpp"
#include "../cpp/double16x16x16.hpp"
#include "../cpp/unittest.hpp"

namespace cl_cpd
{
	extern ContextQueue *cq;
	extern Double16x16x16UnMapped *f;
	extern Double16x16x16BufferFactory *b;

	class Parameter
	{
	public:
		virtual bool validate(const mxArray* input) = 0;
		virtual void setVal(const mxArray* input) = 0;
	};

	template<typename T>
	class XParameter: Parameter
	{
	public:
		void setVal(const mxArray* input)
		{
			val = convert(input);
		}
		virtual T convert(const mxArray* input) = 0;
		T val;
	};

	class BoolParameter: public XParameter<bool>
	{
	public:
		bool validate(const mxArray* input);
		bool convert(const mxArray* input);
	};

	class DoubleParameter: public XParameter<double>
	{
	public:
		bool validate(const mxArray* input);
		double convert(const mxArray* input);
	};

	class StringParameter: public XParameter<std::string>
	{
	public:
		bool validate(const mxArray* input);
		std::string convert(const mxArray* input);
	};

	class TParameter: public XParameter<T<double> >
	{
	public:
		bool validate(const mxArray* input);
		T<double> convert(const mxArray* input);
	};

	class UParameter: public XParameter<U<double> >
	{
	public:
		bool validate(const mxArray* input);
		U<double> convert(const mxArray* input);
	};

	class SumConverter
	{
	public:
		mxArray* convert(const Sum<double>* input);
	};

	class Command
	{
	public:
		virtual std::string getString() = 0;
		virtual std::vector<Parameter*> getParameters() = 0;
		virtual std::vector<mxArray*> handle() = 0;
		virtual ~Command() {};
	};

	class TestCommand: public Command
		{
		TParameter t;
		UParameter u;
		DoubleParameter f;
		DoubleParameter deltaF;
		UParameter g;
		DoubleParameter deltaG;

	public:
		std::string getString()
		{
			return "test";
		}

		std::vector<Parameter*> getParameters()
		{
			std::vector<Parameter*> r(6);
			r[0] = (Parameter*)&t;
			r[1] = (Parameter*)&u;
			r[2] = (Parameter*)&f;
			r[3] = (Parameter*)&deltaF;
			r[4] = (Parameter*)&g;
			r[5] = (Parameter*)&deltaG;
			return r;
		}

		std::vector<mxArray*> handle();
	};

	class InitCommand: public Command
	{
		BoolParameter profile;
	public:
		std::string getString()
		{
			return "init";
		}

		std::vector<Parameter*> getParameters()
		{
			std::vector<Parameter*> r(1);
			r[0] = (Parameter*)&profile;
			return r;
		}

		std::vector<mxArray*> handle();

		~InitCommand()
		{}
	};

	class SetTCommand: public Command
	{
		TParameter t;
		UParameter u;
	public:
		std::string getString()
		{
			return "setTAndU";
		}

		std::vector<Parameter*> getParameters()
		{
			std::vector<Parameter*> r(2);
			r[0] = (Parameter*)&t;
			r[1] = (Parameter*)&u;
			return r;
		}

		std::vector<mxArray*> handle();
	};

	class SetUCommand: public Command
	{
		UParameter u;
	public:
		std::string getString()
		{
			return "setU";
		}

		std::vector<Parameter*> getParameters()
		{
			std::vector<Parameter*> r(1);
			r[0] = (Parameter*)&u;
			return r;
		}

		std::vector<mxArray*> handle();
	};

	class RunCommand: public Command
	{
	public:
		std::string getString()
		{
			return "run";
		}
		std::vector<Parameter*> getParameters()
		{
			return std::vector<Parameter*>(0);
		}

		std::vector<mxArray*> handle();
	};

	class TimeCommand: public Command
	{
	public:
		std::string getString()
		{
			return "time";
		}
		std::vector<Parameter*> getParameters()
		{
			return std::vector<Parameter*>(0);
		}

		std::vector<mxArray*> handle();
	};

}


#endif /* CL_CPD_COMMAND_HPP_ */
