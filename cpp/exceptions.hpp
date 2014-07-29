/*
 * exceptions.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef CL_CPD_EXCEPTIONS_HPP_
#define CL_CPD_EXCEPTIONS_HPP_

#include <sstream>

namespace cl_cpd
{
	class ClCpdException: public std::exception
	{
	protected:
		virtual void setMsg(std::stringstream& ss) const = 0;
	public:
		virtual const char* what() const throw()
		{
			std::stringstream ss;
			this->setMsg(ss);
			return ss.str().c_str();
		}
	};

	class NoPlatformFoundException: public ClCpdException
	{
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "No OpenCL platform found";
		}
	};
	class NoDeviceFoundException: public ClCpdException
	{
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "No OpenCL device found";
		}
	};
	class InvalidSizeOfIException: public ClCpdException
	{
		size_t multiple;
	public:
		InvalidSizeOfIException(size_t multiple) : multiple(multiple){}
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "The length of every dimension of I must be a multiple of ";
			ss << multiple;
		}
	};
	class InvalidSizeOfUException: public ClCpdException
	{
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "Invalid Size Of U Exception";
		}
	};
	class SizesTandUDontMatchException: public ClCpdException
	{
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "The sizes of T and U don't match";
		}
	};
	class SizesUDontMatchException: public ClCpdException
	{
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "SizesUDontMatchException: u.I.size() != u.Us.size()";
		}
	};
	class KernelFileNotFoundException: public ClCpdException
	{
		const char* name;
	public:
		KernelFileNotFoundException(const char* name):name(name){}
	protected:
		void setMsg(std::stringstream& ss) const
		{
			ss << "OpenCL kernel file ";
			ss << name;
			ss << " not found.";
		}
	};
	class InvalidKernelNameException: public ClCpdException
	{
		const char* name;
	public:
		InvalidKernelNameException(const char* name):name(name){}
		void setMsg(std::stringstream& ss) const
		{
			ss << "No kernel with name ";
			ss << name;
			ss << " found.";
		}
	};
}

#endif /* CL_CPD_EXCEPTIONS_HPP_ */
