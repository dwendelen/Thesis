#include "kernelRegister.hpp"

namespace cl_cpd
{
	void KernelHandler::compile()
	{
		if(!compiled)
		{
			kernel->compile();
			compiled = true;
		}
	}

	void KernelRegister::add(cl_cpd::Kernel* kernel)
	{
		KernelHandler kh(kernel);
		m[kh->getName()] = kh;
	}
	cl_cpd::Kernel* KernelRegister::get(std::string name)
	{
		if(m.count(name) == 0)
			return NULL;

		return m[name].getKernel();
	}
	KernelRegister::~KernelRegister()
	{
		for(std::map<std::string, cl_cpd::KernelHandler>::iterator it = m.begin(); it != m.end(); it++)
		{
			delete it->second.getKernel();
		}
	}

}
