# Directory level 1 makefile. This makefile is only used for recursive builds
# at the root level. For directories deeper in the tree, simply update the line
# below to include an extra "../"
include ../Makefile.inc

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
DIRS = $(wildcard */)
OBJLIBSTEM = $(DIRS:%/=%.a)
CURRDIR = $(notdir $(CURDIR))
CURRDIRLIB = $(CURRDIR:%=../lib%.a)

# Default target by convention is ``main''
main: $(CURRDIR) $(OBJLIBSTEM)

# Create library of current directory files
$(CURRDIR): $(OBJECTS)
	$(ECHO) $(AR) $(ARFLAGS) rv $(CURRDIRLIB) $?
	$(AR) $(ARFLAGS) rv $(CURRDIRLIB) $?
	$(ECHO) $(RANLIB) $(CURRDIRLIB)
	$(RANLIB) $(CURRDIRLIB)

# Call subdirectory makefiles to create those libraries
%.a: % force_look
	$(ECHO) Building library from subdirectory $(CURRDIR)$< : $(MAKE) $(MFLAGS)
	cd $<; $(MAKE) $(MFLAGS)

# Suffix rule to convert .cpp -> .o
%.o: %.cpp
	$(ECHO) $(CXX) $(CXXFLAGS) $< -o $@
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	$(ECHO) Cleaning up in subdir $(CURRDIR)
	-rm -f $(OBJECTS)
	-for d in $(DIRS); do (cd $$d; $(MAKE) $(MFLAGS) clean ); done

force_look:
	true
