/*
 * exceptions.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef CL_CPD_EXCEPTIONS_HPP_
#define CL_CPD_EXCEPTIONS_HPP_

namespace cl_cpd
{
	class ClCpdException: public std::exception {};

	class NoPlatformFoundException: public ClCpdException {};
	class NoDeviceFoundException: public ClCpdException {};
	class InvalidSizeOfIException: public ClCpdException {};
	class InvalidSizeOfUException: public ClCpdException {};

}

#endif /* CL_CPD_EXCEPTIONS_HPP_ */
