/*
 * double16x16x16.hpp
 *
 *  Created on: 27 jul 2014
 *      Author: xtrit
 */

#ifndef DOUBLE16X16X16R_HPP_
#define DOUBLE16X16X16R_HPP_

#include "common.hpp"

namespace cl_cpd {

	class Double16x16x16ReMapped: public AbstractFKernel
	{
	public:
		Double16x16x16ReMapped(ContextQueue* cq): AbstractFKernel(cq) {}

	protected:
		u_int getnbDoublesPerWorkitem()
		{
			return 4;
		}

		std::string getFile()
		{
			return "double16x16x16R";
		}
	};

}

#endif /* DOUBLE16X16X16R_HPP_ */
