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
		Kernel(ContextQueue* cq, std::string file):
			cq(cq), kernel(NULL), nanoTime(0), file(file) {}
		void compile();
		void run();
		double getExecutionTimeLastRun();
		virtual ~Kernel();

	protected:
		std::string getCode();

	protected:
		virtual cl::NDRange getLocalSize() = 0;
		virtual cl::NDRange getGlobalSize() = 0;
		cl::Kernel* getKernel();

	private:
		ContextQueue* cq;
		cl::Kernel* kernel;
		double nanoTime;
		std::string file;
	};



	class BlockKernel: public Kernel
	{
	public:
		BlockKernel(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			Kernel(cq, file), I(NULL), nbDoublesPerWorkitem(nbDoublesPerWorkitem) {}
		void setT(cl::Buffer* T);
		void setI(std::vector<size_t>* I);
		bool isValidSizedI(std::vector<size_t>* I);
	protected:
		cl::NDRange getLocalSize();
		cl::NDRange getGlobalSize();

	private:
		std::vector<size_t>* I;
		u_int nbDoublesPerWorkitem;
	};

	class AbstractFKernel: public BlockKernel
	{
	public:
		AbstractFKernel(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			BlockKernel(cq, file, nbDoublesPerWorkitem) {}
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
		AbstractBufferFactory(ContextQueue* cq, u_int nbDoublesPerWorkitem):
			cq(cq), t(NULL), r(0), u(NULL), i(NULL), sum(NULL),
			nbElementsInSum(0), nbDoublesPerWorkitem(nbDoublesPerWorkitem){}

		virtual void init(T t, U u);
		void updateU(U u);
		void readSum(Sum sumArray);

		cl::Buffer* getT(){return t;}
		cl_int getR(){return r;}
		std::vector<cl::Buffer*>* getU(){return u;}
		std::vector<size_t>* getI(){return i;}
		cl::Buffer* getSum(){return sum;}
		size_t getNbElementsInSum(){return nbElementsInSum;}

		virtual ~AbstractBufferFactory();
	protected:
		cl::Buffer* createInitBuf(size_t nbBytes, void* p);
		cl::Buffer* createReadWriteBuf(size_t nbBytes);
		virtual void cleanUp();
		ContextQueue* cq;
	private:
		cl::Buffer* t;
		cl_int r;
		std::vector<cl::Buffer*>* u;
		std::vector<size_t>* i;
		cl::Buffer* sum;
		size_t nbElementsInSum;
		u_int nbDoublesPerWorkitem;

		void cleanUp();
	};

	class AbstractFGBufferFactory:
			public AbstractBufferFactory
	{
	public:
		AbstractFGBufferFactory(ContextQueue* cq, u_int nbDoublesPerWorkitem):
					AbstractBufferFactory(cq, nbDoublesPerWorkitem),
					r(NULL), g(NULL){}
		void init(T t, U u);
		virtual ~AbstractFGBufferFactory();
		void readG(U g);
	protected:
		virtual void cleanUp();
	private:
		cl::Buffer* r;
		std::vector<cl::Buffer*>* g;
	};

#endif /* CL_CPD_COMMON_HPP_ */

