#ifndef CL_CPD_UNITTEST_HPP_
#define CL_CPD_UNITTEST_HPP_

#include "common.hpp"
#include "double16x16x16.hpp"
#include "double16x16x16I.hpp"
#include "double16x16x16R.hpp"
#include "double8x8x8I.hpp"
#include "double8x8x8R.hpp"

namespace cl_cpd
{
	class UnitTest
	{
	public:
		bool test(T<double> t, U<double> u, double f, double delta);
	};
}

#endif /* CL_CPD_UNITTEST_HPP_ */

