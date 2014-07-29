/*
 * cl_cpd_gateway.hpp
 *
 *  Created on: 28 jul 2014
 *      Author: xtrit
 */

#ifndef CL_CPD_GATEWAY_HPP_
#define CL_CPD_GATEWAY_HPP_

#include "mex.h"
#include <new>
#include <map>
#include <string>
#include <sstream>

#include "command.hpp"

struct cmp_str
{
   bool operator()(char const *a, char const *b) const
   {
      return std::strcmp(a, b) < 0;
   }
};

class CommandRegister
{
public:
	CommandRegister() : m() {}
	void add(cl_cpd::Command * c);
	cl_cpd::Command* get(std::string command);
	~CommandRegister();
private:
	std::map<std::string, cl_cpd::Command*> m;
};

void mexFunction(int nbOutput, mxArray *outputArray[],
    int nbInput, const mxArray *inputArray[]);
CommandRegister* buildCommandRegister();


cl_cpd::Command* getCommand(CommandRegister* cr, int nbInput, const mxArray *inputArray[]);

void validateAndFillInput
	(cl_cpd::Command * command, int nbInput, const mxArray * inputArray[]);

void fillOutputArray(int nbOutput, mxArray *outputArray[], std::vector<mxArray *> output);
#endif /* CL_CPD_GATEWAY_HPP_ */
