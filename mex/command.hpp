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

namespace cl_cpd
{
	extern ContextQueue *cq;
	extern Double16x16x16UnMapped *f;
	extern Double16x16x16BufferFactory *b;

	class Converter
	{
	public:
		Converter(){}
		virtual bool validate(const mxArray* input) = 0;
		virtual ~Converter() {};
	};

	class BoolConverter: public Converter
	{
	public:
		BoolConverter() : Converter(){}
		bool validate(const mxArray* input);
		bool convert(const mxArray* input);
		~BoolConverter(){}
	};

	class CStringConverter: public Converter
	{
	public:
		bool validate(const mxArray* input);
		char* convert(const mxArray* input);
		~CStringConverter(){}
	};

	class TConverter: public Converter
	{
	public:
		bool validate(const mxArray* input);
		T* convert(const mxArray* input);
		~TConverter(){}
	};

	class UConverter: public Converter
	{
	public:
		bool validate(const mxArray* input);
		U* convert(const mxArray* input);
		~UConverter(){}
	};

	class SumConverter
	{
	public:
		mxArray* convert(const Sum* input);
	};

	class Command
	{
	public:
		virtual std::string getString() = 0;
		std::vector<Converter*> getConverters() {return converters;}
		virtual std::vector<mxArray*> handle(std::vector<const mxArray*>) = 0;
		virtual ~Command() {};
	protected:
		std::vector<Converter*> converters;
	};

	class InitCommand: public Command
	{
	public:
		InitCommand()
		{
			converters = std::vector<Converter*>(1);
			converters[0] = new BoolConverter();
		}

		std::string getString()
		{
			return "init";
		}

		std::vector<mxArray*> handle(std::vector<const mxArray*>);
		~InitCommand()
		{
			delete converters[0];
		}
	};

	class SetTCommand: public Command
	{
	public:
		SetTCommand()
		{
			converters = std::vector<Converter*>(2);
			converters[0] = new TConverter();
			converters[1] = new UConverter();
		}

		std::string getString()
		{
			return "setT";
		}

		std::vector<mxArray*> handle(std::vector<const mxArray*>);

		~SetTCommand()
		{
			delete converters[0];
			delete converters[1];
		}
	};

	class RunCommand: public Command
	{
	public:
		RunCommand()
		{
			converters = std::vector<Converter *>(0);
		}

		std::string getString()
		{
			return "run";
		}

		std::vector<mxArray*> handle(std::vector<const mxArray*>);

		~RunCommand() {}
	};


}


#endif /* CL_CPD_COMMAND_HPP_ */
