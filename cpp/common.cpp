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


cl::Buffer* AbstractBufferFactory::createInitBuf(size_t nbBytes, void* p)
{
	return new cl::Buffer(*cq->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbBytes, p);
}

cl::Buffer* AbstractBufferFactory::createReadWriteBuf(size_t nbBytes)
{
	return new cl::Buffer(*cq->getContext(), CL_MEM_READ_WRITE, nbBytes);
}

void AbstractBufferFactory::init(T t, U u)
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

	cleanUp();

	size_t s = sizeof(double) * t.I[0] * t.I[1] * t.I[2];
	this->t = createInitBuf(s, t.Ts);

	this->rank = (cl_int) u.rank;

	this->u = new std::vector<cl::Buffer *>(3);

	(*this->u)[0] = createInitBuf(u.size(0), u.Us[0]);
	(*this->u)[1] = createInitBuf(u.size(1), u.Us[1]);
	(*this->u)[2] = createInitBuf(u.size(2), u.Us[2]);

	this->i = new std::vector<size_t>(t.I);

	s = (t.I[0]*t.I[1]*t.I[2])/(4*nbDoublesPerWorkitem*4*nbDoublesPerWorkitem*4*nbDoublesPerWorkitem);
	this->sum = createReadWriteBuf(sizeof(double) * s);

	nbElementsInSum = s;
}



void AbstractBufferFactory::updateU(U u)
{
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[0],CL_FALSE, 0, u.size(0), u.Us[0]);
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[1],CL_FALSE, 0, u.size(1), u.Us[1]);
	cq->getQueue()->enqueueWriteBuffer(*(*this->u)[2],CL_FALSE, 0, u.size(2), u.Us[2]);
}

void AbstractBufferFactory::readSum(Sum sumArray)
{
	cq->getQueue()->enqueueReadBuffer(*sum, CL_TRUE, 0,
			sumArray.nbElements*sizeof(double), sumArray.sum);
}

#define delNull(x) {delete x; x = NULL;}

void AbstractBufferFactory::cleanUp()
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

AbstractBufferFactory::~AbstractBufferFactory()
{
	delete t;
	delete (*u)[0];
	delete (*u)[1];
	delete (*u)[2];
	delete u;
	delete i;
	delete sum;
}

void AbstractFGBufferFactory::init(T t, U u)
{
    AbstractBufferFactory::init(t, u);
    
    size_t s = sizeof(double) * t.I[0] * t.I[1] * t.I[2];
    
    r = createReadWriteBuf(s);

    g = new std::vector<cl::Buffer *>(3);
    (*g)[0] = createReadWriteBuf(u.size(0));
    (*g)[1] = createReadWriteBuf(u.size(1));
    (*g)[2] = createReadWriteBuf(u.size(2));
}

void AbstractFGBufferFactory::readG(U g)
{
	cq->getQueue()->enqueueReadBuffer(*(*this->g)[0], CL_TRUE, 0,
			g.size(0), g.Us[0]);
	cq->getQueue()->enqueueReadBuffer(*(*this->g)[1], CL_TRUE, 0,
				g.size(1), g.Us[1]);
	cq->getQueue()->enqueueReadBuffer(*(*this->g)[2], CL_TRUE, 0,
				g.size(2), g.Us[2]);
}

void AbstractFGBufferFactory::cleanUp()
{
	AbstractBufferFactory::cleanUp();

	delNull(r);
	if(g != NULL)
	{
		delNull((*g)[0]);
		delNull((*g)[1]);
		delNull((*g)[2]);
		delNull(g);
	}
}

AbstractFGBufferFactory::~AbstractFGBufferFactory()
{
	delete r;
	if(g != NULL)
	{
		delete (*g)[0];
		delete (*g)[1];
		delete (*g)[2];
	}
	delete g;
}

void Kernel::compile()
{
	string c = this->getCode();
	cl::Program::Sources s;
	s.push_back(make_pair(c.c_str(), c.length()));
	cl::Program p(*cq->getContext(), s);
	try{
	p.build(*cq->getDevice(), "-cl-opt-disable -g -s \"../opencl/double16x16x16G.cl\"");
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
string Kernel::getCode()
{
	string f = "../opencl/" + file + ".cl";
	ifstream ifs (f.c_str());

	if(!ifs.good())
		throw KernelFileNotFoundException(f.c_str());

	string content( (std::istreambuf_iterator<char>(ifs) ),
	                       (std::istreambuf_iterator<char>()) );

	return content;
}
void Kernel::run()
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
std::vector<double> Kernel::getExecutionTimesLastRun()
{
	return nanoTimes;
}

double Kernel::getExecutionTimeLastRun()
{
	double sum = 0;
	for(std::vector<double>::iterator it = nanoTimes.begin(); it < nanoTimes.end(); ++it)
		sum += *it;

	return sum;
}

template <typename T>
void Kernel::setArg(cl_uint index, T value)
{
	for(std::vector<cl::Kernel*>::iterator it = kernels.begin();
						it < kernels.end(); ++it)
		(*it)->setArg(index, value);
}

Kernel::~Kernel()
{
	for(std::vector<cl::Kernel*>::iterator it = kernels.begin();
					it < kernels.end(); ++it)
		delete *it;
}
/*cl::Kernel* Kernel::getKernel()
{
	return kernel;
}*/

void BlockKernel::setT(cl::Buffer* T)
{
	setArg(0, *T);
}

void BlockKernel::setI(vector<size_t>* I)
{
	if(!isValidSizedI(I))
		throw InvalidSizeOfIException(nbDoublesPerWorkitem * 4);

	this->I = I;
}

bool BlockKernel::isValidSizedI(vector<size_t>* I)
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

cl::NDRange BlockKernel::getLocalSize()
{
	return cl::NDRange(4, 4, 4);
}

std::vector<cl::NDRange> BlockKernel::getGlobalSize()
{
	std::vector<cl::NDRange> v;
	v.push_back(cl::NDRange((*I)[0]/nbDoublesPerWorkitem,(*I)[1]/nbDoublesPerWorkitem,(*I)[2]/nbDoublesPerWorkitem));
	return v;
}

void AbstractFKernel::setRank(cl_int rank)
{
	setArg(4, rank);
}

void AbstractFKernel::setU(vector<cl::Buffer*>* U)
{
	if(!hasUValidNbOfDims(U))
		throw InvalidSizeOfUException();

	setArg(1, *(*U)[0]);
	setArg(2, *(*U)[1]);
	setArg(3, *(*U)[2]);
}

bool AbstractFKernel::hasUValidNbOfDims(std::vector<cl::Buffer*>* U)
{
	return U->size() == 3;
}

void AbstractFKernel::setSum(cl::Buffer* sum)
{
	setArg(5, *sum);
}

cl::NDRange AbstractGKernel::getLocalSize()
{
	return cl::NDRange(8,8);
}
std::vector<cl::NDRange> AbstractGKernel::getGlobalSize()
{
	std::vector<cl::NDRange> v;
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[0]/2));
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[1]/2));
	v.push_back(cl::NDRange((size_t)rank, (size_t)(*I)[2]/2));
	return v;
}

void AbstractGKernel::setR(cl::Buffer* R)
{
	setArg(0, *R);
}
void AbstractGKernel::setU(std::vector<cl::Buffer*>* U)
{
	if(!hasUOrGValidNbOfDims(U))
		throw InvalidSizeOfUException();

	getKernels()[0]->setArg(2, *(*U)[1]);
	getKernels()[0]->setArg(3, *(*U)[2]);

	getKernels()[1]->setArg(1, *(*U)[0]);
	getKernels()[1]->setArg(3, *(*U)[2]);

	getKernels()[2]->setArg(1, *(*U)[0]);
	getKernels()[2]->setArg(2, *(*U)[1]);
}
void AbstractGKernel::setI(std::vector<size_t>* I)
{
	if(!isValidSizedI(I))
		throw InvalidSizeOfIException(16);

	this->I = I;

	getKernels()[0]->setArg(4, (cl_int)(*I)[1]/2);
	getKernels()[0]->setArg(5, (cl_int)(*I)[2]/2);

	getKernels()[1]->setArg(4, (cl_int)(*I)[0]/2);
	getKernels()[1]->setArg(5, (cl_int)(*I)[2]/2);

	getKernels()[2]->setArg(4, (cl_int)(*I)[0]/2);
	getKernels()[2]->setArg(5, (cl_int)(*I)[1]/2);
}

bool AbstractGKernel::isValidSizedI(vector<size_t>* I)
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
bool AbstractGKernel::hasUOrGValidNbOfDims(std::vector<cl::Buffer*>* UorG)
{
	return UorG->size() == 3;
}

void AbstractGKernel::setG(std::vector<cl::Buffer*>* G)
{
	if(!hasUOrGValidNbOfDims(G))
		throw InvalidSizeOfUException();

	getKernels()[0]->setArg(1, *(*G)[0]);
	getKernels()[1]->setArg(2, *(*G)[1]);
	getKernels()[2]->setArg(3, *(*G)[2]);
}
void AbstractGKernel::setRank(cl_int rank)
{
	this->rank = rank;
}

void AbstractTMapper::setTMapped(cl::Buffer* TMapped)
{
	setArg(1, TMapped);
}


