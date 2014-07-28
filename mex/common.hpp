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
#include <fstream>
#include <iostream>

#include "cl.hpp"
#include "exceptions.hpp"

namespace cl_cpd
{
	struct T
	{
		std::vector<size_t> I;
		double* Ts;
	};

	struct U
	{
		size_t R;
		std::vector<size_t> I;
		std::vector<double*> Us;

		size_t size(size_t dim)
		{
			return sizeof(double) * R * I[dim];
		}
	};

	struct Sum
	{
		size_t nbElements;
		double* sum;
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
		Kernel(ContextQueue* cq): cq(cq), kernel(NULL), nanoTime(0) {}
		void compile();
		void run();
		uint64_t getExecutionTimeLastRun();
		virtual ~Kernel();

	protected:
		std::string getCode();
		virtual std::string getFile() = 0;

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
		BlockKernel(ContextQueue* cq): Kernel(cq), I(NULL) {}
		void setT(cl::Buffer* T);
		void setI(std::vector<size_t>* I);
		bool isValidSizedI(std::vector<size_t>* I);
	protected:
		virtual u_int getnbDoublesPerWorkitem() = 0;
		cl::NDRange getLocalSize();
		cl::NDRange getGlobalSize();

	private:
		std::vector<size_t>* I;
	};

	class AbstractFKernel: public BlockKernel
	{
	public:
		AbstractFKernel(ContextQueue* cq): BlockKernel(cq) {}
		void setR(cl_int R);
		void setU(std::vector<cl::Buffer*>* U);
		bool hasUValidNbOfDims(std::vector<cl::Buffer*>* U);
		void setSum(cl::Buffer* sum);

	};

	class AbstractTMapper: public BlockKernel
	{
	public:
		void setTMapped(cl::Buffer* TMapped);
	};

	class AbstractBufferFactory
	{
	public:
		AbstractBufferFactory(ContextQueue* cq):
			cq(cq), t(NULL), r(0), u(NULL), i(NULL), sum(NULL), sumArray(NULL){}

		void init(T t, U u);
		void updateU(U u);
		void readSum();

		cl::Buffer* getT(){return t;}
		cl_int getR(){return r;}
		std::vector<cl::Buffer*>* getU(){return u;}
		std::vector<size_t>* getI(){return i;}
		cl::Buffer* getSum(){return sum;}
		Sum* getSumArray(){return sumArray;}

		virtual ~AbstractBufferFactory();
	protected:
		cl::Buffer* createInitBuf(size_t nbBytes, void* p);
		cl::Buffer* createReadWriteBuf(size_t nbBytes);
		virtual u_int getnbDoublesPerWorkitem() = 0;
	private:
		ContextQueue* cq;
		cl::Buffer* t;
		cl_int r;
		std::vector<cl::Buffer*>* u;
		std::vector<size_t>* i;
		cl::Buffer* sum;
		Sum* sumArray;

		void cleanUp();
	};

}

#endif /* CL_CPD_COMMON_HPP_ */

