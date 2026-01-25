#ifndef POT_BRIDGE_H
#define POT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the client context
typedef struct PotClient PotClient;

/**
 * Initialize the RPC client.
 * Returns NULL on failure. Check pot_get_last_error() for details.
 */
PotClient *pot_client_init(const char *host, int32_t port);

/**
 * Free the client resources. Safe to call with NULL.
 */
void pot_client_free(PotClient *client);

/**
 * Execute the potential calculation.
 * * @param client The opaque client handle.
 * @param natoms Number of atoms (matches Int32 in schema).
 * @param pos    Array of positions [3 * natoms] (flattened x,y,z).
 * @param atmnrs Array of atomic numbers [natoms].
 * @param box    Array of box vectors [9] (row-major 3x3).
 * @param out_energy Pointer to write the resulting energy.
 * @param out_forces Buffer to write resulting forces [3 * natoms].
 * * @return 0 on success, non-zero on failure.
 */
int32_t pot_calculate(PotClient *client, int32_t natoms, const double *pos,
                      const int32_t *atmnrs, const double *box,
                      double *out_energy, double *out_forces);

/**
 * Retrieve the last error message for this client.
 * Returns an empty string if no error occurred.
 */
const char *pot_get_last_error(PotClient *client);

#ifdef __cplusplus
}
#endif

#endif // POT_BRIDGE_H
