/*
 * res.hpp
 *
 *  Created on: 21 aug 2014
 *      Author: xtrit
 */

#ifndef RES_HPP_
#define RES_HPP_


#include "common.hpp"

namespace cl_cpd
{
	struct line
	{
		std::string name;
		std::vector<double> x;
		std::vector<double> y;
	};
	struct graph
	{
		int val;
		std::vector<line> lines;
	};


	void invlR(std::vector<graph>& gs);
	void fl16Pack(std::vector<graph>& gs);

	void fl8Pack(std::vector<graph>& gs);

	void fl816(std::vector<graph>& gs);
	void fl816Pack(std::vector<graph>& gs);

}


#endif /* RES_HPP_ */
