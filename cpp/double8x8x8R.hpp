/*
 * double16x16x16.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef DOUBLE8X8X8R_HPP_
#define DOUBLE8X8X8R_HPP_

#include "common.hpp"

namespace cl_cpd {

	class Double8x8x8ReMapped: public AbstractFKernel
	{
	public:
		Double8x8x8ReMapped(ContextQueue* cq):
			AbstractFKernel(cq, "double8x8x8R", 2) {}
	};

	class Double8x8x8BufferFactory: public AbstractBufferFactory
	{
	public:
		Double8x8x8BufferFactory(ContextQueue* cq):
			AbstractBufferFactory(cq, 2){}
	};
}

#endif /* DOUBLE8X8X8R_HPP_ */
