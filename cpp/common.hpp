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
	template<typename type>
	struct T
	{
		std::vector<size_t> I;
		type* Ts;
	};

	template<typename type>
	struct U
	{
		size_t rank;
		std::vector<size_t> I;
		std::vector<type*> Us;

		size_t size(size_t dim)
		{
			return sizeof(type) * rank * I[dim];
		}
	};

	template<typename type>
	struct Sum
	{
		size_t nbElements;
		type* sum;
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

	template<typename type>
	class BufferFactory
	{
	public:
		BufferFactory(ContextQueue* cq): cq(cq), t(NULL), rank(0), u(NULL), i(NULL), sum(NULL),
				nbElementsInSum(0){}
		void updateU(U<type> u);

		virtual cl::Buffer* getT(){return t;}
		cl_int getRank(){return rank;}
		std::vector<cl::Buffer*>* getU(){return u;}
		std::vector<size_t>* getI(){return i;}
		cl::Buffer* getSum(){return sum;}
		size_t getNbElementsInSum(){return nbElementsInSum;}
		void readSum(Sum<type> sumArray);

		virtual ~BufferFactory();
	protected:
		cl::Buffer* createInitBuf(size_t nbBytes, void* p);
		cl::Buffer* createReadWriteBuf(size_t nbBytes);
		virtual void cleanUp();
		ContextQueue* cq;
		cl::Buffer* t;
		cl_int rank;
		std::vector<cl::Buffer*>* u;
		std::vector<size_t>* i;
		cl::Buffer* sum;
		size_t nbElementsInSum;
	};

	template<typename type>
	class AbstractBufferFactory: public BufferFactory<type>
	{
	public:
		AbstractBufferFactory(ContextQueue* cq, u_int nbDoublesPerWorkitem):
			BufferFactory<type>(cq), nbDoublesPerWorkitem(nbDoublesPerWorkitem){}
		virtual void init(T<type> t, U<type> u);
		virtual ~AbstractBufferFactory(){}
	private:
		u_int nbDoublesPerWorkitem;
	};

	template<typename type>
	class AbstractMappedBufferFactory: public AbstractBufferFactory<type>
	{
		private:
		cl::Buffer* tMapped;
	public:
		AbstractMappedBufferFactory(ContextQueue* cq, u_int nbDoublesPerWorkitem):
			AbstractBufferFactory<type>(cq, nbDoublesPerWorkitem), tMapped(NULL){}
		virtual void init(T<type> t, U<type> u);
		cl::Buffer* getTUnMapped(){return AbstractBufferFactory<type>::getT();}
		virtual cl::Buffer* getT(){return tMapped;}
		virtual ~AbstractMappedBufferFactory(){delete tMapped;}

	};

	template<typename type>
	class OneDRangeBufferFactory: public BufferFactory<type>
	{

	public:
		OneDRangeBufferFactory(ContextQueue* cq, u_int nbWorkitems):
			BufferFactory<type>(cq), nbWorkitems(nbWorkitems){}
		virtual void init(T<type> t, U<type> u);
		cl::LocalSpaceArg getL() {return L;}
		virtual ~OneDRangeBufferFactory(){}
	protected:
		u_int nbWorkitems;
		cl::LocalSpaceArg L;
	};

	template<typename type>
	class AbstractFGBufferFactory:
			public AbstractBufferFactory<type>
	{
	public:
		AbstractFGBufferFactory(ContextQueue* cq, u_int nbDoublesPerWorkitem):
					AbstractBufferFactory<type>(cq, nbDoublesPerWorkitem),
					f(NULL), g(NULL){}
		void init(T<type> t, U<type> u);
		cl::Buffer* getF(){return f;}
		std::vector<cl::Buffer*>* getG(){return g;}

		void readG(U<type> g);
		virtual ~AbstractFGBufferFactory();
	protected:
		virtual void cleanUp();
	private:
		cl::Buffer* f;
		std::vector<cl::Buffer*>* g;
	};

	template<typename type>
	class Kernel
	{
	public:
		Kernel<type>(ContextQueue* cq, std::string file):
			cq(cq), file(file){}
		void compile();
		void run();
		std::vector<double> getExecutionTimesLastRun();
		std::string getName() {return file;}
		double getExecutionTimeLastRun();
		virtual ~Kernel();
		virtual void setBuffers(BufferFactory<type>* b) = 0;

	protected:
		std::string getCode();
		std::vector<cl::Kernel*> getKernels() {return kernels;}
		template <typename T>
		void setArg(cl_uint index, T value);

	protected:
		virtual cl::NDRange getLocalSize() = 0;
		virtual std::vector<cl::NDRange> getGlobalSize() = 0;
		std::vector<std::string> kernelNames;
	private:
		ContextQueue* cq;
		std::vector<cl::Kernel*> kernels;
		std::vector<double> nanoTimes;
		std::string file;

	};

	/*template<typename type>
		class FKernel: public virtual Kernel<type>
	{
	public:
		FKernel(){}
		virtual void setBuffers(BufferFactory<type>* b) = 0;
		virtual ~FKernel(){}
	};*/

	template<typename type>
	class OneDRangeKernel: public Kernel<type>
	{
		std::vector<size_t>* I;
		u_int nbWorkitems;
	public:
		OneDRangeKernel(ContextQueue* cq, std::string file, u_int nbWorkitems):
			Kernel<type>(cq, file), I(NULL), nbWorkitems(nbWorkitems)
			{
				Kernel<type>::kernelNames.push_back("Kernel");
			}

		virtual cl::NDRange getLocalSize();
		virtual std::vector<cl::NDRange> getGlobalSize();
		void setT(cl::Buffer* T);
		void setRank(cl_int rank);
		void setU(std::vector<cl::Buffer*>* U);
		void setI(std::vector<size_t>* I);
		void setSum(cl::Buffer* sum);
		void setL(cl::LocalSpaceArg L);

		virtual void setBuffers(BufferFactory<type>* b)
		{
			setT(b->getT());
			setRank(b->getRank());
			setU(b->getU());
			setI(b->getI());
			setSum(b->getSum());
		}
		virtual ~OneDRangeKernel(){}
	};

	template<typename type>
	class BlockKernel: public Kernel<type>
	{
	public:
		BlockKernel(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			Kernel<type>(cq, file), I(NULL), nbDoublesPerWorkitem(nbDoublesPerWorkitem){}
		void setT(cl::Buffer* T);
		void setI(std::vector<size_t>* I);
		bool isValidSizedI(std::vector<size_t>* I);
		virtual ~BlockKernel(){}
	protected:
		cl::NDRange getLocalSize();
		virtual std::vector<cl::NDRange> getGlobalSize();

	private:
		std::vector<size_t>* I;
		u_int nbDoublesPerWorkitem;
	};

	template<typename type>
	class AbstractFKernel: public BlockKernel<type>
	{
	public:
		AbstractFKernel(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			BlockKernel<type>(cq, file, nbDoublesPerWorkitem)
		{
			BlockKernel<type>::kernelNames.push_back("Kernel");
		}
		void setRank(cl_int R);
		void setU(std::vector<cl::Buffer*>* U);
		bool hasUValidNbOfDims(std::vector<cl::Buffer*>* U);
		void setSum(cl::Buffer* sum);
		virtual void setBuffers(BufferFactory<type>* b)
		{
			setT(b->getT());
			setRank(b->getRank());
			setU(b->getU());
			setI(b->getI());
			setSum(b->getSum());
		}
		virtual ~AbstractFKernel(){}
	};

	template<typename type>
	class AbstractFGKernel: public AbstractFKernel<type>
	{
	public:
		AbstractFGKernel(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			AbstractFKernel<type>(cq, file, nbDoublesPerWorkitem)
		{}
		void setF(cl::Buffer* F);
		virtual void setBuffers(AbstractFGBufferFactory<type>* b)
		{
			AbstractFKernel<type>::setBuffers(b);
			setF(b->getF());
		}
		virtual ~AbstractFGKernel(){};
	};

	//Uses T as G
	template<typename type>
	class AbstractGKernel: public Kernel<type>
	{
	public:
		AbstractGKernel(ContextQueue* cq, std::string file):
			Kernel<type>(cq, file), I(NULL), rank(0)
		{
			Kernel<type>::kernelNames.clear();
			Kernel<type>::kernelNames.push_back("KernelG1");
			Kernel<type>::kernelNames.push_back("KernelG2");
			Kernel<type>::kernelNames.push_back("KernelG3");
		}
		cl::NDRange getLocalSize();
		virtual std::vector<cl::NDRange> getGlobalSize();

		void setF(cl::Buffer* F);
		void setU(std::vector<cl::Buffer*>*);
		void setI(std::vector<size_t>* I);
		void setG(std::vector<cl::Buffer*>*);
		void setRank(cl_int rank);

		bool hasUOrGValidNbOfDims(std::vector<cl::Buffer*>* UorG);
		bool isValidSizedI(std::vector<size_t>* I);

		virtual void setBuffers(BufferFactory<type>* b){}

		virtual void setBuffers(AbstractFGBufferFactory<type>* b)
		{
			setF(b->getF());
			setU(b->getU());
			setI(b->getI());
			setG(b->getG());
			setRank(b->getRank());

		}

		virtual ~AbstractGKernel(){}
	private:
		std::vector<size_t>* I;
		cl_int rank;
	};

	template<typename type>
	class AbstractTMapper: public BlockKernel<type>
	{
	public:
		AbstractTMapper(ContextQueue* cq, std::string file, u_int nbDoublesPerWorkitem):
			BlockKernel<type>(cq, file, nbDoublesPerWorkitem){}
		void setTMapped(cl::Buffer* TMapped);
		virtual void setBuffers(BufferFactory<type>* b)
		{
			setI(b->getI());
			setTMapped(b->getT());
		}
		virtual void setBuffers(AbstractMappedBufferFactory<type>* b)
		{
			setT(b->getTUnMapped());
			setI(b->getI());
			setTMapped(b->getT());
		}

		virtual ~AbstractTMapper(){}
	};
}
#endif /* CL_CPD_COMMON_HPP_ */

