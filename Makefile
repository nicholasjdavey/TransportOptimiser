include Makefile.inc

# The dependency tree is only to two subdirectory levels. We have opted for a
# non-recursive make as we do not intend to build libraries
SOURCES = $(wildcard */*.cpp wildcard */*/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# If we are doing a library build (libraries to only two levels of dependency)
DIRS = $(wildcard */)
CURRSOURCES = $(wildcard *.cpp)
CURROBJS = $(CURRSOURCES:.cpp=.o)
OBJLIBSTEM = $(DIRS:%/=%.a)
OBJLIBS = $(DIRS:%/=lib%.a)
LIBS = -L. $(wildcard *.a)

# Main executable
MAINFILE = TransOpt.cpp
EXE = $(MAINFILE:.cpp=.exe)

# Default is an incremental linking of dependencies
$(EXE): $(OBJECTS) $(CURROBJS) force_look
	$(ECHO) $(CXX) $(CXXFLAGS) $(OBJECTS) $(CURROBJS) -o $(EXE)
	$(LD) $(OBJECTS) $(CURROBJS) $(LXXFLAGS) -o $(EXE)

# Build for creating and linking libraries
withlibs: $(LIBS) $(CURROBJS) force_look
	$(ECHO) $(LD) -o $(EXE) $(CURROBJS) $(LIBS)
	$(LD) -o $(EXE) $(CURROBJS) $(LIBS)

# For creating libraries using recursive makefiles
makelibs: $(OBJLIBSTEM) force_look

# Creating a single top-level library
%.a: % force_look
	$(ECHO) Building library from directory $< : $(MAKE) $(MFLAGS)
	cd $<; $(MAKE) $(MFLAGS)

# Suffix rule to convert .cpp -> .o
%.o: %.cpp
	$(ECHO) $(CXX) $(CXXFLAGS) $< -o $@
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-rm $(OBJECTS) $(CURROBJS) $(EXE)

cleanlibs:
	-rm $(CURROBJS) $(EXE)
	-for d in $(DIRS); do (cd $$d; $(MAKE) $(MFLAGS) clean ); done

help:
	@echo ""
	@echo "make		- builds TransportOptimiser using incremental linking"
	@echo "make libraries	- builds all subfolders of the project as separate libraries"
	@echo "make withlibs	- recursive application build (N.B. run make libraries first)"
	@echo "make clean	- remove all *.o files and executables in all subdirectories"
	@echo "make cleanlibs	- cleans a recursive build"
	@echo "make help	- this info"
	@echo ""

force_look :
	true
