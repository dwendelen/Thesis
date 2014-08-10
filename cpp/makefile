
RM := rm -rf


# All Target
all: meas

# Tool invocations
meas: 
	g++ -I/opt/AMDAPP/include meas.cpp common.cpp -lOpenCL -o meas
# Other Targets
clean:
	-$(RM) meas

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
