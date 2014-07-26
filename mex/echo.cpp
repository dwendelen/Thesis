#include "echo.hpp"

class Echo
{
    public:
    int i;
};



void v()
{
    Echo * e = new Echo();
    e->i = 5;
    delete e;
}
