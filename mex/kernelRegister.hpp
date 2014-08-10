#ifndef CL_CPD_KREG_HPP_
#define CL_CPD_KREG_HPP_

#include "../cpp/common.hpp"

namespace cl_cpd
{
	class KernelHandler
	{
		bool compiled;
		cl_cpd::Kernel* kernel;
	public:
		KernelHandler(cl_cpd::Kernel* kernel): compiled(false), kernel(kernel){}
		void compile();
		std::string getName(){return kernel->getName();}
		cl_cpd::Kernel* getKernel(){return kernel;}
	};

	class KernelRegister
	{
		std::map<std::string, cl_cpd::KernelHandler> m;
	public:
		void add(cl_cpd::Kernel* kernel);
		cl_cpd::Kernel* get(std::string name);
		~KernelRegister();
	};
}
#endif /* CL_CPD_KREG_HPP_ */
