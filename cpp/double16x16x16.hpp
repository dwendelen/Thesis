/*
 * double16x16x16.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef DOUBLE16X16X16_HPP_
#define DOUBLE16X16X16_HPP_

#include "common.hpp"

namespace cl_cpd {

	class Double16x16x16UnMapped: public AbstractFKernel
	{
	public:
		Double16x16x16UnMapped(ContextQueue* cq):
			AbstractFKernel(cq, "double16x16x16", 4) {}
	};

	class Double16x16x16BufferFactory: public AbstractBufferFactory
	{
	public:
		Double16x16x16BufferFactory(ContextQueue* cq):
			AbstractBufferFactory(cq, 4){}
	};

	class Double16x16x16G: public AbstractGKernel
	{
	public:
		Double16x16x16G(ContextQueue* cq):
			AbstractGKernel(cq, "double16x16x16G"){}
	};

	class Double16x16x16FGBufferFactory: public AbstractFGBufferFactory
	{
	public:
		Double16x16x16FGBufferFactory(ContextQueue* cq):
			AbstractFGBufferFactory(cq, 4){}
	};

}

#endif /* DOUBLE16X16X16_HPP_ */
