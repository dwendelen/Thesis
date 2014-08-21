#include "cl_cpd_gateway.hpp"

using namespace cl_cpd;

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
void* operator new[](std::size_t size) throw(std::bad_alloc)
{
	//mexPrintf("new[]");
	return operator new(size);
}
void operator delete[](void* ptr) throw()
{
	//mexPrintf("delete[]");
	operator delete(ptr);
}
void* operator new [](std::size_t size, const std::nothrow_t&) throw()
{
	//mexPrintf("delete[]");
	return operator new(size, std::nothrow_t());
}
void operator delete[](void* ptr, const std::nothrow_t&) throw()
{
	//mexPrintf("delete[]");
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

void mexFunction(int nbOutput, mxArray *outputArray[],
    int nbInput, const mxArray *inputArray[])
    {
		mexAtExit(clean);
		try{
			CommandRegister* cr = buildCommandRegister();
			Command* command = getCommand(cr, nbInput, inputArray);
			std::cout << "validateAndFillInput";
			validateAndFillInput(command, nbInput, inputArray);
			std::cout << "handle";
			std::vector<mxArray *> output = command->handle();
			std::cout << "output";
			fillOutputArray(nbOutput, outputArray, output);
		}
		catch (cl::Error &e)
		{
			std::stringstream ss;
			ss << "Exception OpenCL: " << e.what() << " code: " << e.err();
			mexErrMsgTxt(ss.str().c_str());
		}
		catch (ClCpdException &e)
		{
			mexErrMsgTxt(e.what());
		}
    }

CommandRegister* buildCommandRegister()
{
	CommandRegister* cr = new CommandRegister();
	cr->add(new InitCommand);
	cr->add(new SetTCommand);
	cr->add(new SetUCommand);
	cr->add(new RunCommand);
	cr->add(new TimeCommand);
	cr->add(new TestCommand);
	cr->add(new InvlRCommand);

	return cr;
}

Command* getCommand(CommandRegister* cr, int nbInput, const mxArray *inputArray[])
{
	if(nbInput < 1)
		mexErrMsgTxt("No command supplied.");

	StringParameter p;
	p.setVal(inputArray[0]);
	Command* c = cr->get(p.val);

	if(c == NULL)
	{
		std::stringstream ss;
		ss << "Command " << p.val << " is unknown.";
		mexErrMsgTxt(ss.str().c_str());
	}

	return c;
}

void validateAndFillInput(Command* command, int nbInput, const mxArray * inputArray[])
{
	std::cout << 0;
	if(((size_t)nbInput) < command->getParameters().size() + 1)
	{
		std::cout << 0.5;
		std::stringstream ss;
		ss << command->getString() << " requires ";
		ss << command->getParameters().size() << " parameters, not ";
		ss << (nbInput - 1) << ".";
		mexErrMsgTxt(ss.str().c_str());
	}

	for(mwIndex i = 0; i < command->getParameters().size(); i++)
	{
		std::cout << 2;
		if(!command->getParameters()[i]->validate(inputArray[i + 1]))
		{
			std::stringstream ss;
			ss << "Parameter " << i << " is invalid.";
			mexErrMsgTxt(ss.str().c_str());
		}
		std::cout << 3;
		command->getParameters()[i]->setVal(inputArray[i + 1]);
	}
}

void fillOutputArray(int nbOutput, mxArray *outputArray[], std::vector<mxArray *> output)
{
	for(size_t i = 0; i < output.size() && i < (size_t)nbOutput; i++)
	{
		outputArray[i] = output[i];
	}
}
