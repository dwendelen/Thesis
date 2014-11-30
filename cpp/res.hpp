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

	struct Data
	{
		std::vector<int> R;
		std::vector<int> I;
		int nbKernels;
		//data[R][I][kernel]
		double* data;

		size_t size()
		{
            std::cout << "Nb Elements in data\n";
            std::cout << R.size() * I.size() * nbKernels << "\n";
			return R.size() * I.size() * nbKernels;
		}
	};

	void invlR(std::vector<graph>& gs);
	void measureF(Data& data);
	void measureG(Data& data);
}


#endif /* RES_HPP_ */
