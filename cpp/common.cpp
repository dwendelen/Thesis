/*
 * common.cpp
 *
 *  Created on: 26 jul 2014
 *      Author: xtrit
 */

#include "common.hpp"

using namespace std;
using namespace cl_cpd;

ContextQueue::ContextQueue() :
		profile(false),
		context(NULL),
		queue(NULL),
		device(NULL){}

void ContextQueue::init()
{
	init(false);
}

void ContextQueue::init(bool profile)
{
	this->profile = profile;

	vector<cl::Platform> ps;
	vector<cl::Device> ds;
	cl::Platform::get(&ps);

	if(ps.size() == 0)
		throw NoPlatformFoundException();

	try {
		ps[0].getDevices(CL_DEVICE_TYPE_GPU, &ds);
	}
	catch (cl::Error& e) {
		if(e.err() != CL_DEVICE_NOT_FOUND)
			throw e;
	}

	if(ds.size() == 0)
		try {
			ps[0].getDevices(CL_DEVICE_TYPE_CPU, &ds);
		} catch (cl::Error& e) {
			if(e.err() != CL_DEVICE_NOT_FOUND)
				throw e;
		}


	if(ds.size() == 0)
		throw NoDeviceFoundException();

	//Only use the first device
	ds.resize(1);
	this->context = new cl::Context(ds);
	this->device = new vector<cl::Device>(ds);

	cl_command_queue_properties props = 0;

	if(profile)
		props = CL_QUEUE_PROFILING_ENABLE;

	this->queue = new cl::CommandQueue(*this->context, ds[0], props);
}

cl::Context* ContextQueue::getContext()
{
	return this->context;
}

cl::CommandQueue* ContextQueue::getQueue()
{
	return this->queue;
}

vector<cl::Device>* ContextQueue::getDevice()
{
	return this->device;
}

bool ContextQueue::isProfile()
{
	return this->profile;
}

ContextQueue::~ContextQueue()
{
	delete context;
	delete queue;
	delete device;
}

template<typename type>
cl::Buffer* BufferFactory<type>::createInitBuf(size_t nbBytes, void* p)
{
	return new cl::Buffer(*cq->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbBytes, p);
}

template<typename type>
cl::Buffer* BufferFactory<type>::createReadWriteBuf(size_t nbBytes)
{
	return new cl::Buffer(*cq->getContext(), CL_MEM_READ_WRITE, nbBytes);
}

template<typename type>
void AbstractBufferFactory<type>::init(T<type> t, U<type> u)
{
	if(t.I.size() != u.I.size())
		throw SizesTandUDontMatchException();

	if(u.I.size() != 3)
		throw InvalidSizeOfUException();

	if(!equal(t.I.begin(), t.I.end(), u.I.begin()))
		throw SizesTandUDontMatchException();

	if(u.I.size() != u.Us.size())
		throw SizesUDontMatchException();

	if(t.I[0] % (nbDoublesPerWorkitem * 4) != 0)
			throw InvalidSizeOfIException(nbDoublesPerWorkitem * 4);

	if(t.I[1] % (nbDoublesPerWorkitem * 4) != 0)
			throw InvalidSizeOfIException(nbDoublesPerWorkitem * 4);

	if(t.I[2] % (nbDoublesPerWorkitem * 4) != 0)
			throw InvalidSizeOfIException(nbDoublesPerWorkitem * 4);

	BufferFactory<type>::cleanUp();

	size_t s = sizeof(type) * t.I[0] * t.I[1] * t.I[2];
	this->t = createInitBuf(s, t.Ts);

	this->rank = (cl_int) u.rank;

	this->u = new std::vector<cl::Buffer *>(3);

	(*this->u)[0] = createInitBuf(u.size(0), u.Us[0]);
	(*this->u)[1] = createInitBuf(u.size(1), u.Us[1]);
	(*this->u)[2] = createInitBuf(u.size(2), u.Us[2]);

	this->i = new std::vector<size_t>(t.I);

	s = (t.I[0]*t.I[1]*t.I[2])/(4*nbDoublesPerWorkitem*4*nbDoublesPerWorkitem*4*nbDoublesPerWorkitem);
	BufferFactory<type>::sum = BufferFactory<type>::createReadWriteBuf(sizeof(type) * s);

	BufferFactory<type>::nbElementsInSum = s;
}

template<typename type>
void AbstractMappedBufferFactory<type>::init(T<type> t, U<type> u)
{
	AbstractBufferFactory<type>::init(t, u);

	size_t s = sizeof(type) * t.I[0] * t.I[1] * t.I[2];

	delete tMapped;
	tMapped = AbstractBufferFactory<type>::createReadWriteBuf(s);
}


template<typename type>
void OneDRangeBufferFactory<type>::init(T<type> t, U<type> u)
{
	if(t.I.size() != u.I.size())
		throw SizesTandUDontMatchException();

	if(u.I.size() != 3)
		throw InvalidSizeOfUException();

	if(!equal(t.I.begin(), t.I.end(), u.I.begin()))
		throw SizesTandUDontMatchException();

	if(u.I.size() != u.Us.size())
		throw SizesUDontMatchException();

	if(t.I[0]*t.I[1]*t.I[2] % nbWorkitems != 0)
		throw InvalidSizeOfIException(nbWorkitems);

	BufferFactory<type>::cleanUp();

	size_t s = sizeof(type) * t.I[0] * t.I[1] * t.I[2];
	this->t = createInitBuf(s, t.Ts);

	this->rank = (cl_int) u.rank;

	this->u = new std::vector<cl::Buffer *>(3);

	(*this->u)[0] = createInitBuf(u.size(0), u.Us[0]);
	(*this->u)[1] = createInitBuf(u.size(1), u.Us[1]);
	(*this->u)[2] = createInitBuf(u.size(2), u.Us[2]);

	this->i = new std::vector<size_t>(t.I);

	this->L = cl::__local(sizeof(type) * nbWorkitems);

	s = (t.I[0]*t.I[1]*t.I[2])/nbWorkitems;
	BufferFactory<type>::sum = BufferFactory<type>::createReadWriteBuf(sizeof(type) * s);

	BufferFactory<type>::nbElementsInSum = s;
}

template<typename type>
void BufferFactory<type>::updateU(U<type> u)
{
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[0],CL_FALSE, 0, u.size(0), u.Us[0]);
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[1],CL_FALSE, 0, u.size(1), u.Us[1]);
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[2],CL_FALSE, 0, u.size(2), u.Us[2]);
}

template<typename type>
void BufferFactory<type>::readSum(Sum<type> sumArray)
{
	cq->getQueue()->enqueueReadBuffer(*sum, CL_TRUE, 0,
			sumArray.nbElements*sizeof(type), sumArray.sum);
}

#define delNull(x) {delete x; x = NULL;}

template<typename type>
void BufferFactory<type>::cleanUp()
{
	delNull(t);
	rank = 0;
	if(u != NULL)
	{
		delNull((*u)[0]);
		delNull((*u)[1]);
		delNull((*u)[2]);
		delNull(u);
	}
	delNull(i);
	delNull(sum);
}

template<typename type>
BufferFactory<type>::~BufferFactory()
{
	delete t;
	delete (*u)[0];
	delete (*u)[1];
	delete (*u)[2];
	delete u;
	delete i;
	delete sum;
}

template<typename type>
void AbstractFGBufferFactory<type>::init(T<type> t, U<type> u)
{
    AbstractBufferFactory<type>::init(t, u);
    
    size_t s = sizeof(type) * t.I[0] * t.I[1] * t.I[2];
    
    f = BufferFactory<type>::createReadWriteBuf(s);

    g = new std::vector<cl::Buffer *>(3);
    (*g)[0] = createReadWriteBuf(u.size(0));
    (*g)[1] = createReadWriteBuf(u.size(1));
    (*g)[2] = createReadWriteBuf(u.size(2));
}

template<typename type>
void AbstractFGBufferFactory<type>::readG(U<type> g)
{
	BufferFactory<type>::cq->getQueue()->enqueueReadBuffer(*(*this->g)[0], CL_TRUE, 0,
			g.size(0), g.Us[0]);
	BufferFactory<type>::cq->getQueue()->enqueueReadBuffer(*(*this->g)[1], CL_TRUE, 0,
				g.size(1), g.Us[1]);
	BufferFactory<type>::cq->getQueue()->enqueueReadBuffer(*(*this->g)[2], CL_TRUE, 0,
				g.size(2), g.Us[2]);
}



template<typename type>
void AbstractFGBufferFactory<type>::cleanUp()
{
	BufferFactory<type>::cleanUp();

	delNull(f);
	if(g != NULL)
	{
		delNull((*g)[0]);
		delNull((*g)[1]);
		delNull((*g)[2]);
		delNull(g);
	}
}

template<typename type>
AbstractFGBufferFactory<type>::~AbstractFGBufferFactory()
{
	delete f;
	if(g != NULL)
	{
		delete (*g)[0];
		delete (*g)[1];
		delete (*g)[2];
	}
	delete g;
}

template<typename type>
void Kernel<type>::compile()
{
	string c = this->getCode();
	cl::Program::Sources s;
	s.push_back(make_pair(c.c_str(), c.length()));
	cl::Program p(*cq->getContext(), s);
	try{
	p.build(*cq->getDevice());
	}catch (cl::Error &e) {
		cout << p.getBuildInfo<CL_PROGRAM_BUILD_LOG>((*cq->getDevice())[0]);
	}
	cout << p.getBuildInfo<CL_PROGRAM_BUILD_LOG>((*cq->getDevice())[0]);

	for(std::vector<std::string>::iterator it = kernelNames.begin();
			it < kernelNames.end(); ++it)
	{
		try {
			this->kernels.push_back(new cl::Kernel(p, (*it).c_str()));
		} catch (cl::Error &e) {
			if(e.err() == CL_INVALID_KERNEL_NAME)
				throw InvalidKernelNameException(*it);
			else
				throw e;
		}
	}
}
template<typename type>
string Kernel<type>::getCode()
{
	string f = "../opencl/" + file + ".cl";
	ifstream ifs (f.c_str());

	if(!ifs.good())
		throw KernelFileNotFoundException(f.c_str());

	string content( (std::istreambuf_iterator<char>(ifs) ),
	                       (std::istreambuf_iterator<char>()) );

	return content;
}
template<typename type>
void Kernel<type>::run()
{
	std::vector<cl::Event> es;
	cl::Event e;

	std::vector<cl::NDRange>* gSize = new std::vector<cl::NDRange>;
	*gSize = getGlobalSize();

	std::vector<cl::NDRange>::iterator it2 = (*gSize).begin();
	//First enqueue
	for(std::vector<cl::Kernel*>::iterator it = kernels.begin();
			it < kernels.end(); ++it, ++it2)
	{
		cq->getQueue()->enqueueNDRangeKernel(**it, cl::NDRange(0, 0, 0), *it2, getLocalSize(),
			NULL, &e);
		es.push_back(e);
	}

	delete gSize;

	//Then start waiting
	if(cq->isProfile())
	{
		nanoTimes.clear();
		for(std::vector<cl::Event>::iterator it = es.begin();
						it < es.end(); ++it)
		{
		(*it).wait();
		nanoTimes.push_back(e.getProfilingInfo<CL_PROFILING_COMMAND_END>()
				- e.getProfilingInfo<CL_PROFILING_COMMAND_START>());
		}
	}

}
template<typename type>
std::vector<double> Kernel<type>::getExecutionTimesLastRun()
{
	return nanoTimes;
}
template<typename type>
double Kernel<type>::getExecutionTimeLastRun()
{
	double sum = 0;
	for(std::vector<double>::iterator it = nanoTimes.begin(); it < nanoTimes.end(); ++it)
		sum += *it;

	return sum;
}

template <typename type>
template <typename T>
void Kernel<type>::setArg(cl_uint index, T value)
{
	for(std::vector<cl::Kernel*>::iterator it = kernels.begin();
						it < kernels.end(); ++it)
		(*it)->setArg(index, value);
}

template<typename type>
Kernel<type>::~Kernel()
{
	for(std::vector<cl::Kernel*>::iterator it = kernels.begin();
					it < kernels.end(); ++it)
		delete *it;
}
/*cl::Kernel* Kernel::getKernel()
{
	return kernel;
}*/
template<typename type>
void OneDRangeKernel<type>::setT(cl::Buffer* T)
{
	Kernel<type>::setArg(0, *T);
}
template<typename type>
void OneDRangeKernel<type>::setRank(cl_int rank)
{
	Kernel<type>::setArg(4, rank);
}

template<typename type>
void OneDRangeKernel<type>::setU(vector<cl::Buffer*>* U)
{
	Kernel<type>::setArg(1, *(*U)[0]);
	Kernel<type>::setArg(2, *(*U)[1]);
	Kernel<type>::setArg(3, *(*U)[2]);
}

template<typename type>
void OneDRangeKernel<type>::setI(vector<size_t>* I)
{
	this->I = I;
	Kernel<type>::setArg(6, (cl_int) (*I)[0]);
	Kernel<type>::setArg(7, (cl_int) (*I)[1]);
	Kernel<type>::setArg(8, (cl_int) (*I)[2]);
}

template<typename type>
void OneDRangeKernel<type>::setL(cl::LocalSpaceArg L)
{
	Kernel<type>::setArg(9, L);
}

template<typename type>
void OneDRangeKernel<type>::setSum(cl::Buffer* sum)
{
	Kernel<type>::setArg(5, *sum);
}

template<typename type>
cl::NDRange OneDRangeKernel<type>::getLocalSize()
{
	return cl::NDRange(nbWorkitems);
}

template<typename type>
std::vector<cl::NDRange> OneDRangeKernel<type>::getGlobalSize()
{
	std::vector<cl::NDRange> v;
	v.push_back(cl::NDRange((*I)[0] * (*I)[1] * (*I)[2]));
	return v;
}

template<typename type>
void BlockKernel<type>::setT(cl::Buffer* T)
{
	Kernel<type>::setArg(0, *T);
}

template<typename type>
void BlockKernel<type>::setI(vector<size_t>* I)
{
	if(!isValidSizedI(I))
		throw InvalidSizeOfIException(nbDoublesPerWorkitem * 4);

	this->I = I;
}

template<typename type>
bool BlockKernel<type>::isValidSizedI(vector<size_t>* I)
{
	if(I->size() != 3)
		return false;

	if((*I)[0] % (nbDoublesPerWorkitem * 4) != 0)
		return false;

	if((*I)[1] % (nbDoublesPerWorkitem * 4) != 0)
			return false;

	if((*I)[2] % (nbDoublesPerWorkitem * 4) != 0)
			return false;

	return true;
}

template<typename type>
cl::NDRange BlockKernel<type>::getLocalSize()
{
	return cl::NDRange(4, 4, 4);
}

template<typename type>
std::vector<cl::NDRange> BlockKernel<type>::getGlobalSize()
{
	std::vector<cl::NDRange> v;
	v.push_back(cl::NDRange((*I)[0]/nbDoublesPerWorkitem,(*I)[1]/nbDoublesPerWorkitem,(*I)[2]/nbDoublesPerWorkitem));
	return v;
}

template<typename type>
void AbstractFKernel<type>::setRank(cl_int rank)
{
	Kernel<type>::setArg(4, rank);
}

template<typename type>
void AbstractFKernel<type>::setU(vector<cl::Buffer*>* U)
{
	if(!hasUValidNbOfDims(U))
		throw InvalidSizeOfUException();

	Kernel<type>::setArg(1, *(*U)[0]);
	Kernel<type>::setArg(2, *(*U)[1]);
	Kernel<type>::setArg(3, *(*U)[2]);
}

template<typename type>
void AbstractFGKernel<type>::setF(cl::Buffer* F)
{
	Kernel<type>::setArg(6, *F);
}

template<typename type>
bool AbstractFKernel<type>::hasUValidNbOfDims(std::vector<cl::Buffer*>* U)
{
	return U->size() == 3;
}

template<typename type>
void AbstractFKernel<type>::setSum(cl::Buffer* sum)
{
	Kernel<type>::setArg(5, *sum);
}

template<typename type>
cl::NDRange AbstractGKernel<type>::getLocalSize()
{
	return cl::NDRange(8,8);
}

template<typename type>
std::vector<cl::NDRange> AbstractGKernel<type>::getGlobalSize()
{
	std::vector<cl::NDRange> v;
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[0]/2));
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[1]/2));
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[2]/2));
	return v;
}

template<typename type>
void AbstractGKernel<type>::setF(cl::Buffer* F)
{
	Kernel<type>::setArg(0, *F);
}

template<typename type>
void AbstractGKernel<type>::setU(std::vector<cl::Buffer*>* U)
{
	if(!hasUOrGValidNbOfDims(U))
		throw InvalidSizeOfUException();

	Kernel<type>::getKernels()[0]->setArg(2, *(*U)[1]);
	Kernel<type>::getKernels()[0]->setArg(3, *(*U)[2]);

	Kernel<type>::getKernels()[1]->setArg(1, *(*U)[0]);
	Kernel<type>::getKernels()[1]->setArg(3, *(*U)[2]);

	Kernel<type>::getKernels()[2]->setArg(1, *(*U)[0]);
	Kernel<type>::getKernels()[2]->setArg(2, *(*U)[1]);
}

template<typename type>
void AbstractGKernel<type>::setI(std::vector<size_t>* I)
{
	if(!isValidSizedI(I))
		throw InvalidSizeOfIException(16);

	this->I = I;

	Kernel<type>::getKernels()[0]->setArg(4, (cl_int)(*I)[1]/2);
	Kernel<type>::getKernels()[0]->setArg(5, (cl_int)(*I)[2]/2);

	Kernel<type>::getKernels()[1]->setArg(4, (cl_int)(*I)[0]/2);
	Kernel<type>::getKernels()[1]->setArg(5, (cl_int)(*I)[2]/2);

	Kernel<type>::getKernels()[2]->setArg(4, (cl_int)(*I)[0]/2);
	Kernel<type>::getKernels()[2]->setArg(5, (cl_int)(*I)[1]/2);
}

template<typename type>
bool AbstractGKernel<type>::isValidSizedI(vector<size_t>* I)
{
	if(I->size() != 3)
		return false;

	if((*I)[0] % 16 != 0)
		return false;

	if((*I)[1] % 16 != 0)
			return false;

	if((*I)[2] % 16 != 0)
			return false;

	return true;
}

template<typename type>
bool AbstractGKernel<type>::hasUOrGValidNbOfDims(std::vector<cl::Buffer*>* UorG)
{
	return UorG->size() == 3;
}


template<typename type>
void AbstractGKernel<type>::setG(std::vector<cl::Buffer*>* G)
{
	if(!hasUOrGValidNbOfDims(G))
		throw InvalidSizeOfUException();

	Kernel<type>::getKernels()[0]->setArg(1, *(*G)[0]);
	Kernel<type>::getKernels()[1]->setArg(2, *(*G)[1]);
	Kernel<type>::getKernels()[2]->setArg(3, *(*G)[2]);
}

template<typename type>
void AbstractGKernel<type>::setRank(cl_int rank)
{
	this->rank = rank;
}

template<typename type>
void AbstractTMapper<type>::setTMapped(cl::Buffer* TMapped)
{
	Kernel<type>::setArg(1, TMapped);
}

template class BufferFactory<double>;
template class BufferFactory<float>;

template class AbstractBufferFactory<double>;
template class AbstractBufferFactory<float>;

template class OneDRangeBufferFactory<double>;
template class OneDRangeBufferFactory<float>;

template class AbstractFGBufferFactory<double>;
template class AbstractFGBufferFactory<float>;

template class AbstractMappedBufferFactory<double>;
template class AbstractMappedBufferFactory<float>;

template class Kernel<double>;
template class Kernel<float>;

template class AbstractFKernel<double>;
template class AbstractFKernel<float>;

template class OneDRangeKernel<float>;
template class OneDRangeKernel<double>;

template class AbstractFGKernel<double>;
template class AbstractFGKernel<float>;

template class AbstractGKernel<double>;
template class AbstractGKernel<float>;

template class AbstractTMapper<double>;
template class AbstractTMapper<float>;
