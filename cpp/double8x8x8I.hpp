/*
 * double16x16x16.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef DOUBLE8X8X8I_HPP_
#define DOUBLE8X8X8I_HPP_

#include "common.hpp"

namespace cl_cpd {

	class Double8x8x8Isolated: public AbstractFKernel<double>
	{
	public:
		Double8x8x8Isolated(ContextQueue* cq):
			AbstractFKernel(cq, "double8x8x8I", 2) {}
	};
}

#endif /* DOUBLE8X8X8I_HPP_ */
