`LJPot` keys the neighbour list it keeps on each thread on
`Neighbours::generation()` instead of its own address.
A destructor releases only the lists of the thread it runs on, so an `LJPot`
built at the address of a destroyed one used to find that one's list on every
other thread.
