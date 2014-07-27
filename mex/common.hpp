/*
 * common.hpp
 *
 *  Created on: 26 jul 2014
 *      Author: xtrit
 */

#ifndef CL_CPD_COMMON_HPP_
#define CL_CPD_COMMON_HPP_

#include <string>
#include <exception>

#include "cl.hpp"
#include "exceptions.hpp"

namespace cl_cpd
{
	struct T
	{
		std::vector<size_t> I;
		double* T;
	};

	struct U
	{
		size_t R;
		std::vector<size_t> I;
		std::vector<double*> U;
	};

	class ContextQueue
	{
	public:
		ContextQueue();
		void init();
		void init(bool profile);
		cl::Context* getContext();
		cl::CommandQueue* getQueue();
		std::vector<cl::Device>* getDevice();
		bool isProfile();
		~ContextQueue();

	private:
		bool profile;
		cl::Context* context;
		cl::CommandQueue* queue;
		std::vector<cl::Device>* device;
	};

	class Kernel
	{
	public:
		Kernel(ContextQueue* cq);
		void compile();
		void run();
		uint64_t getExecutionTimeLastRun();
		virtual ~Kernel();

	protected:
		virtual std::string getCode() = 0;

	protected:
		virtual cl::NDRange getLocalSize() = 0;
		virtual cl::NDRange getGlobalSize() = 0;
		cl::Kernel* getKernel();

	private:
		ContextQueue* cq;
		cl::Kernel* kernel;
		uint64_t nanoTime;
	};



	class BlockKernel: public Kernel
	{
	public:
		void setT(cl::Buffer T);
		void setI(std::vector<size_t> I);
		bool isValidSizedI(std::vector<size_t> I);
	protected:
		virtual u_int getnbDoublesPerWorkitem() = 0;
		cl::NDRange getLocalSize();
		cl::NDRange getGlobalSize();

	private:
		std::vector<size_t> I;
	};

	class AbstractFKernel: public BlockKernel
	{
	public:
		void setR(cl_int R);
		void setU(std::vector<cl::Buffer> U);
		bool hasUValidNbOfDims(std::vector<cl::Buffer> U);
		void setSum(cl::Buffer sum);

	};

	class AbstractTMapper: public BlockKernel
	{
	public:
		void setTMapped(cl::Buffer TMapped);
	};

	class AbstractBufferFactory
	{
	public:
		AbstractBufferFactory(ContextQueue cq) : cq(cq){}

	private:
		ContextQueue cq;
	};

}

#endif /* CL_CPD_COMMON_HPP_ */

