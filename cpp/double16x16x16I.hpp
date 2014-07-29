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

	class Double16x16x16Isolated: public AbstractFKernel
	{
	public:
		Double16x16x16Isolated(ContextQueue* cq): AbstractFKernel(cq) {}

	protected:
		u_int getnbDoublesPerWorkitem()
		{
			return 4;
		}

		std::string getFile()
		{
			return "double16x16x16I";
		}
	};

}

#endif /* DOUBLE16X16X16I_HPP_ */
