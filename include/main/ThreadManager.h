#ifndef THREADMANAGER_H
#define THREADMANAGER_H

// For now the code runs on a single machine with shared memory

typedef std::shared_ptr<ctpl::thread_pool> PoolPtr;

class ThreadManager;
typedef std::shared_ptr<ThreadManager> ThreadManagerPtr;

/**
 * @brief The ThreadManager class
 *
 * Class for managing threads for the Road Design software using thread pools
 */
class ThreadManager {

public:

// CONSTRUCTORS ///////////////////////////////////////////////////////////////
    ThreadManager(unsigned long max_threads);

// ACCESSORS //////////////////////////////////////////////////////////////////
    unsigned long getMaxThreads() {
        return ThreadManager::max_threads;
    }

    unsigned long getNoThreads() {
        return ThreadManager::noThreads;
    }

// STATIC ROUTINES ////////////////////////////////////////////////////////////

// CALCULATION ROUTINES ///////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
private:
    unsigned long max_threads;   /**< User-defined number of threads */
    unsigned long noThreads;     /**< Actual number of threads */
    PoolPtr pool;             /**< Pool for managing threads */
};
#endif
