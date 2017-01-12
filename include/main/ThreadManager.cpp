#include "../transportbase.h"

ThreadManager::ThreadManager(unsigned long max_threads) {
    unsigned long const hardware_threads = std::thread::hardware_concurrency();

    // The actual number of threads is the minimum of what the system can normally
    // handle and what we want
    this->noThreads = std::min(hardware_threads != 0 ? hardware_threads : 2,
            max_threads);

    PoolPtr pool(new ctpl::thread_pool(this->noThreads));
    this->pool = pool;
}

ThreadManager::~ThreadManager(){
}
