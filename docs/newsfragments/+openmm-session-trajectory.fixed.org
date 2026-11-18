The OpenMM MD session takes its first Langevin force on the hyperplane, keeps
its random stream across calls, holds serialized distance constraints, and
reflects at walls until the state is inside, where it used to return after
eight passes.
