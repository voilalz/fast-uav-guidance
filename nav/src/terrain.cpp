#include "nav/terrain.hpp"

namespace nav {

bool Dem::height(real n, real e, real& h) const {
  if (!contains(n, e)) return false;
  const real fr = n / sp_, fc = e / sp_;
  const std::size_t r = static_cast<std::size_t>(fr), c = static_cast<std::size_t>(fc);
  const real tr = fr - r, tc = fc - c;
  h = (1 - tr) * ((1 - tc) * at(r, c) + tc * at(r, c + 1)) + tr * ((1 - tc) * at(r + 1, c) + tc * at(r + 1, c + 1));
  return true;
}

bool Dem::gradient(real n, real e, real& gn, real& ge) const {
  real hn1, hn0, he1, he0;
  const real d = sp_;
  if (!height(n + d, e, hn1) || !height(n - d, e, hn0) || !height(n, e + d, he1) || !height(n, e - d, he0))
    return false;
  gn = (hn1 - hn0) / (2 * d);
  ge = (he1 - he0) / (2 * d);
  return true;
}

}  // namespace nav
