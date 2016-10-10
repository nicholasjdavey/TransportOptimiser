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
    /**
     * Returns the user-input desired number of threads
     *
     * @return User-requested number of threads as unsigned long
     */
    unsigned long getMaxThreads() {
        return ThreadManager::max_threads;
    }

    /**
     * Returns the actual number of independent threads in the pool
     *
     * @return Number of independent threads in pool as unsigned long
     */
    unsigned long getNoThreads() {
        return ThreadManager::noThreads;
    }

    /**
     * Returns the thread pool
     *
     * @return Thread pool as PoolPtr
     */
    PoolPtr getPool() {
        return this->pool;
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
