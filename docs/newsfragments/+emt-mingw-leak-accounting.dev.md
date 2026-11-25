The EMT leak-accounting test now pairs each counted deallocation with its
allocation, so it stays valid when MinGW routes a delete across runtime modules.
