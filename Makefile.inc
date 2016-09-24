CXX = g++
CC = gcc
LD = g++
AR = ar
ARFLAGS =
RANLIB = ranlib
DEBUG = -g
ECHO = echo
LDFLAGS = -I C:/MinGW/include/cppLibraries/eigen -I C:/MinGW/include/cppLibraries/ -I C:/MinGW/include/cppLibraries/boost_1_58_0 -I C:/MinGW/include/cppLibraries/libigl-master/include/igl
CXXFLAGS = -std=gnu++11 -pthread -c $(LDFLAGS) $(LDLIBS) $(DEBUG)
LXXFLAGS = -std=gnu++11 -pthread $(LDFLAGS) $(LDLIBS) $(DEBUG)

SHELL = /bin/sh

# Clear out all suffixes
#.SUFFIXES:
# List only those we use
#.SUFFIXES: .o .cpp

.SILENT: