#include "cl_cpd_gateway.hpp"

using namespace cl_cpd;

//scalar, throwing new and it matching delete
void* operator new (std::size_t n) throw(std::bad_alloc)
{
    //mexPrintf("New");
    void* p = mxMalloc(n);
    mexMakeMemoryPersistent(p);
    if(p == NULL)
        throw std::bad_alloc();

    return p;
}
void operator delete (void* p) throw()
{
    //mexPrintf("Del");
    mxFree(p);
}
//scalar, nothrow new and it matching delete
void* operator new (std::size_t n,const std::nothrow_t&) throw()
{
    void* p = mxMalloc(n);
    mexMakeMemoryPersistent(p);
    return p;
} 
void operator delete (void* p, const std::nothrow_t&) throw()
{
    //mexPrintf("Del");
    mxFree(p);
}

//array throwing new and matching delete[]
void* operator new[](std::size_t size) throw(std::bad_alloc)
{
	mexPrintf("new[]");
	return operator new(size);
}
void operator delete[](void* ptr) throw()
{
	mexPrintf("delete[]");
	operator delete(ptr);
}

//array, nothrow new and matching delete[]
void* operator new [](std::size_t size, const std::nothrow_t&) throw()
{
	mexPrintf("delete[]");
	return operator new(size, std::nothrow_t());
}
void operator delete[](void* ptr, const std::nothrow_t&) throw()
{
	mexPrintf("delete[]");
	operator delete[](ptr, std::nothrow_t());
}


void CommandRegister::add(Command* c)
{
	std::cout << "Added " << c->getString();
	m[c->getString()] = c;
}

CommandRegister::~CommandRegister()
{
	for(std::map<std::string, cl_cpd::Command*>::iterator it = m.begin(); it != m.end(); it++)
	{
		delete it->second;
	}
}

Command* CommandRegister::get(std::string command)
{
	if(m.count(command) == 0)
		return NULL;

	return m[command];
}

void clean()
{
	delete f;
	delete b;
	delete cq;
}

void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
    {
		mexAtExit(clean);
		try{
			CommandRegister* cr = buildCommandRegister();
			Command* command = getCommand(cr, nrhs, prhs);
			std::vector<const mxArray *> input = validateAndVectoriseParameters(command, nrhs, prhs);
			std::vector<mxArray *> output = command->handle(input);
		}
		catch (cl::Error &e)
		{
			std::stringstream ss;
			ss << "Exception OpenCL: " << e.what() << " code: " << e.err();
			mexErrMsgTxt(ss.str().c_str());
		}
    }

CommandRegister* buildCommandRegister()
{
	CommandRegister* cr = new CommandRegister();
	cr->add(new InitCommand);
	cr->add(new SetTCommand);
	cr->add(new RunCommand);

	return cr;
}

Command* getCommand(CommandRegister* cr, int nrhs, const mxArray *prhs[])
{
	if(nrhs < 1)
		mexErrMsgTxt("No command supplied.");

	char* commandStr = CStringConverter().convert(prhs[0]);
	Command* c = cr->get(commandStr);

	if(c == NULL)
	{
		std::stringstream ss;
		ss << "Command " << commandStr << " is unknown.";
		mexErrMsgTxt(ss.str().c_str());
	}

	return c;
}

std::vector<const mxArray *> validateAndVectoriseParameters(Command* command, int nrhs, const mxArray * prhs[])
{
	std::cout << 0;
	if(((size_t)nrhs) < command->getConverters().size() + 1)
	{
		std::cout << 0.5;
		std::stringstream ss;
		ss << command->getString() << " requires ";
		ss << command->getConverters().size() << " parameters, not ";
		ss << (nrhs - 1) << ".";
		mexErrMsgTxt(ss.str().c_str());
	}

	std::cout << 1;

	std::vector<const mxArray *> input;

	for(mwIndex i = 0; i < command->getConverters().size(); i++)
	{
		std::cout << 2;
		if(!command->getConverters()[i]->validate(prhs[i + 1]))
		{
			std::stringstream ss;
			ss << "Parameter " << i << " is invalid.";
			mexErrMsgTxt(ss.str().c_str());
		}
		std::cout << 3;
		input.push_back(prhs[i + 1]);
	}

	return input;
}
