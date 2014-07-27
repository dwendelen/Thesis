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

class double16x16x16UnMapped: public AbstractFKernel
{
	u_int getnbDoublesPerWorkitem()
	{
		return 4;
	}

	std::string getCode()
	{
		return "";
	}
};

}

#endif /* DOUBLE16X16X16_HPP_ */
