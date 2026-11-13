#include "rgpot/SuttonChen/SuttonChenKernel.hpp"

extern "C" void rgpot_sc_bench(const double *pos, std::size_t n, double epsilon,
                               double c, double a, int n_exp, int m_exp,
                               double cutoff, double *forces, double *energy) {
  rgpot::sutton_chen_direct(pos, n, epsilon, c, a, n_exp, m_exp, cutoff, forces,
                            energy, nullptr);
}
