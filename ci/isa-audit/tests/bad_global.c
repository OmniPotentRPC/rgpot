/* Compiled with -mavx2 -O3. The integer add vectorises to vpaddd (AVX2)
   in an unlisted function. */
__attribute__((noinline)) void hot_loop(int *out, const int *a, const int *b,
                                         int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}
