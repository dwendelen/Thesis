/*
 * double16x16x16.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef DOUBLE16X16X16I_HPP_
#define DOUBLE16X16X16I_HPP_

#include "common.hpp"

namespace cl_cpd {

	class Double16x16x16Isolated: public AbstractFKernel<double>
	{
	public:
		Double16x16x16Isolated(ContextQueue* cq):
			AbstractFKernel(cq, "double16x16x16I", 4) {}
	};

}

#endif /* DOUBLE16X16X16I_HPP_ */
