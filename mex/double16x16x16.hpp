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
		Double16x16x16UnMapped(ContextQueue* cq) : AbstractFKernel(cq) {}

	protected:
		u_int getnbDoublesPerWorkitem()
		{
			return 4;
		}

		std::string getFile()
		{
			return "double16x16x16";
		}
	};


	class Double16x16x16BufferFactory: public AbstractBufferFactory
	{
	public:
		Double16x16x16BufferFactory(ContextQueue* cq):AbstractBufferFactory(cq){}
	protected:
		u_int getnbDoublesPerWorkitem()
		{
			return 4;
		}
	};



}

#endif /* DOUBLE16X16X16_HPP_ */
