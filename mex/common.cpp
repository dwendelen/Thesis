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

	ps[0].getDevices(CL_DEVICE_TYPE_GPU, &ds);

	if(ds.size() == 0)
		ps[0].getDevices(CL_DEVICE_TYPE_CPU, &ds);

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

Kernel::Kernel(ContextQueue* cq) : cq(cq)
{
	this->kernel = NULL;
	this->nanoTime = 0;
}
void Kernel::compile()
{
	string c = this->getCode();
	cl::Program::Sources s;
	s[0] = make_pair(c.c_str(), c.length());
	cl::Program p(*cq->getContext(), s);
	p.build(*cq->getDevice());
	this->kernel = new cl::Kernel(p, "Kernel");
}
void Kernel::run()
{
	cl::Event e;

	cq->getQueue()->enqueueNDRangeKernel(*kernel, cl::NDRange(0, 0, 0), getGlobalSize(), getLocalSize(),
			NULL, &e);

	if(cq->isProfile())
	{
		e.wait();
		nanoTime = e.getProfilingInfo<CL_PROFILING_COMMAND_END>()
				- e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	}

}
uint64_t Kernel::getExecutionTimeLastRun()
{
	return nanoTime;
}
Kernel::~Kernel()
{
	delete kernel;
}
cl::Kernel* Kernel::getKernel()
{
	return kernel;
}

void BlockKernel::setT(cl::Buffer T)
{
	getKernel()->setArg(0, T);
}

void BlockKernel::setI(vector<size_t> I)
{
	if(!isValidSizedI(I))
		throw InvalidSizeOfIException();

	this->I = I;
}

bool BlockKernel::isValidSizedI(vector<size_t> I)
{
	if(I.size() != 3)
		return false;

	if(I[0] % (getnbDoublesPerWorkitem() * 4) != 0)
		return false;

	if(I[1] % (getnbDoublesPerWorkitem() * 4) != 0)
			return false;

	if(I[2] % (getnbDoublesPerWorkitem() * 4) != 0)
			return false;

	return true;
}

cl::NDRange BlockKernel::getLocalSize()
{
	return cl::NDRange(4, 4, 4);
}

cl::NDRange BlockKernel::getGlobalSize()
{
	return cl::NDRange(I[0]/4,I[1]/4,I[2]/4);
}

void AbstractFKernel::setR(cl_int R)
{
	getKernel()->setArg(4, R);
}

void AbstractFKernel::setU(vector<cl::Buffer> U)
{
	if(!hasUValidNbOfDims(U))
		throw InvalidSizeOfUException();

	getKernel()->setArg(1, U[0]);
	getKernel()->setArg(2, U[1]);
	getKernel()->setArg(3, U[2]);
}

bool AbstractFKernel::hasUValidNbOfDims(std::vector<cl::Buffer> U)
{
	return U.size() == 3;
}

void AbstractFKernel::setSum(cl::Buffer sum)
{
	getKernel()->setArg(5, sum);
}

void AbstractTMapper::setTMapped(cl::Buffer TMapped)
{
	getKernel()->setArg(1, TMapped);
}


