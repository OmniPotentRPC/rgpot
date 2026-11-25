#ifndef POT_BRIDGE_H
#define POT_BRIDGE_H

/**
 * @brief C API for the RPC potential client bridge.
 *
 * This header provides a C-compatible interface to the distributed potential
 * calculation client, allowing integration with Fortran or other languages with
 * a C interface, e.g. Julia.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MSVC shared builds need an explicit export or pot_client_bridge.lib is not
 * produced and meson install fails looking for the import library. */
#if defined(_WIN32) || defined(_WIN64)
#  if defined(POT_BRIDGE_BUILD)
#    define POT_BRIDGE_API __declspec(dllexport)
#  else
#    define POT_BRIDGE_API __declspec(dllimport)
#  endif
#else
#  define POT_BRIDGE_API
#endif

/**
 * @brief Opaque handle to the client context.
 */
typedef struct PotClient PotClient;

/**
 * @brief Initializes the RPC client connection.
 * @param host Target server hostname or IP address.
 * @param port Network port of the potential server.
 * @return Pointer to the client context or @c NULL on failure.
 * @see pot_get_last_error
 */
POT_BRIDGE_API PotClient *pot_client_init(const char *host, int32_t port);

/**
 * @brief Frees all resources associated with the client.
 * @param client The opaque client handle to release.
 * @return Void.
 */
POT_BRIDGE_API void pot_client_free(PotClient *client);

/**
 * @brief Select the keyed orbitals used by following calculations.
 * @param client The opaque client handle.
 * @param key Image or bead key.
 * @return 0 on success, non-zero when @a client is invalid.
 */
POT_BRIDGE_API int32_t pot_client_select_orbitals(PotClient *client,
                                                  int64_t key);

/**
 * @brief Executes a remote potential calculation.
 * @pre The @a client must be successfully initialized.
 * @param client The opaque client handle.
 * @param natoms Total number of atoms in the system.
 * @param pos Array of flattened atomic coordinates.
 * @param atmnrs Array of atomic numbers.
 * @param box Simulation cell vectors in row-major order.
 * @param out_energy Pointer to store the calculated energy.
 * @param out_forces Buffer to store the calculated forces.
 * @return 0 on success, non-zero on failure.
 */
POT_BRIDGE_API int32_t pot_calculate(PotClient *client, int32_t natoms, const double *pos,
                      const int32_t *atmnrs, const double *box,
                      double *out_energy, double *out_forces);

/**
 * @brief Request parameters for a remote neighbour-list query.
 */
typedef struct PotNeighborsRequest {
  double cutoff;
  double skin;
  uint8_t format;
  uint8_t cut;
  int32_t periodic[3];
  int32_t shifts;
  int32_t vectors;
  int32_t distances;
} PotNeighborsRequest;

/**
 * @brief Caller-owned output arrays for a remote neighbour-list query.
 */
typedef struct PotNeighborsOut {
  int32_t *first;
  int32_t *second;
  int32_t *shift;
  double *vector;
  double *distance;
  uint8_t *mask;
  size_t cap;
} PotNeighborsOut;

/**
 * @brief Queries the remote potential's neighbour list for a configuration.
 * @param client The opaque client handle.
 * @param natoms Total number of atoms in the system.
 * @param pos Array of flattened atomic coordinates.
 * @param atmnrs Array of atomic numbers.
 * @param box Simulation cell vectors in row-major order.
 * @param request Neighbour query parameters.
 * @param out Caller-allocated buffers, or @c NULL for a size probe.
 * @param out_count Receives the number of pairs written.
 * @param out_needed Receives the total number of pairs in the list.
 * @return 0 on success, 6 when @a out->cap is shorter than @a *out_needed,
 *         or another non-zero code on failure.
 */
POT_BRIDGE_API int32_t pot_neighbours(PotClient *client, int32_t natoms,
                                      const double *pos, const int32_t *atmnrs,
                                      const double *box,
                                      const PotNeighborsRequest *request,
                                      PotNeighborsOut *out, size_t *out_count,
                                      size_t *out_needed);

/**
 * @brief Retrieves the most recent error message.
 * @param client The opaque client handle.
 * @return String containing the error description.
 */
POT_BRIDGE_API const char *pot_get_last_error(PotClient *client);

#ifdef __cplusplus
}
#endif

#endif // POT_BRIDGE_H
