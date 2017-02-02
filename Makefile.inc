#CXX = g++
CXX = nvcc
NVCC = nvcc
CC = gcc
#LD = g++
LD = nvcc
AR = ar
ARFLAGS =
RANLIB = ranlib
DEBUG = -g
DEBUGEXTRA = -rdynamic
ECHO = echo
OS := $(shell uname)

ifeq ($(OS),MINGW32_NT-6.1)
# For compilation using MinGW64 on Windows 7 or later
	CFLAGS = -I C:/MinGW/include/cppLibraries/eigen -I C:/MinGW/include/cppLibraries/ -I C:/MinGW/include/cppLibraries/boost_1_58_0 -I C:/MinGW/include/cppLibraries/libigl-master/include/igl
	LFLAGS = -I C:/MinGW/include/cppLibraries/OpenCV/lib
else
# For compilation under Linux
	CFLAGS = -I/usr/local/include
	LFLAGS = -I/usr/local/lib -I/usr/local/cuda/lib64
endif

LDLIBS = -lpthread -lboost_iostreams -lboost_system -lboost_filesystem -lcurand
# Uncomment the below two lines to enable debugging
#CXXFLAGS = -std=gnu++11 -pthread -c $(CFLAGS) $(DEBUG)
#LXXFLAGS = -std=gnu++11 -pthread $(CFLAGS) $(LFLAGS) $(LDLIBS) $(DEBUG)

# The two lines below enable optimisation
#CXXFLAGS = -std=gnu++11 -pthread -O2 -c $(CFLAGS)
#LXXFLAGS = -std=gnu++11 -pthread -O2 $(CFLAGS) $(LFLAGS) $(LDLIBS)

# Uncomment the below three lines to enable CUDA support
CXXFLAGS = -std=c++11 -c -Wno-deprecated-gpu-targets $(CFLAGS) $(DEBUG)
LXXFLAGS = -std=c++11 -Wno-deprecated-gpu-targets $(CFLAGS) $(LFLAGS) $(LDLIBS) $(DEBUG)
NVCCFLAGS = -std=c++11 -c -Wno-deprecated-gpu-targets $(CFLAGS) $(DEBUG)

SHELL = /bin/sh

# Clear out all suffixes
#.SUFFIXES:
# List only those we use
#.SUFFIXES: .o .cpp

.SILENT:
