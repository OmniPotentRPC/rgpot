/*
 * rgpot Python plugin bridge.
 *
 * This is a generic C wrapper that loads a Python module implementing
 * the rgpot plugin protocol (plugin_create, plugin_destroy, plugin_calculate)
 * and exposes it as a native rgpot plugin .so.
 *
 * The Python module name is specified via the config JSON:
 *   {"module": "harmonic_plugin", "k": 1.0}
 *   {"module": "jax_lj_plugin", "epsilon": 1.0, "sigma": 1.0}
 *   {"module": "torch_nn_plugin", "model_path": "model.pt"}
 *
 * Build:
 *   cc -shared -fPIC -o rgpot_python.so python_bridge.c \
 *      $(python3-config --includes --ldflags --embed) \
 *      -I../../../include
 *
 * Use:
 *   export RGPOT_PLUGIN_PATH=/path/to/dir
 *   export PYTHONPATH=/path/to/python/plugins
 *
 * The bridge calls into the Python module's plugin_create/destroy/calculate
 * functions, marshaling flat C arrays to/from numpy arrays.
 */

#include <rgpot/plugin.h>

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Per-instance state */
typedef struct {
    PyObject *module;
    int py_handle;
    char last_error[512];
} PyPluginInstance;

static int py_initialized = 0;

static void ensure_python(void) {
    if (!py_initialized) {
        Py_Initialize();
        /* Make numpy available */
        PyRun_SimpleString("import numpy");
        py_initialized = 1;
    }
}

static rgpot_plugin_status_t
py_create(const char *config_json, rgpot_plugin_instance **out) {
    ensure_python();

    /* Parse module name from config JSON (minimal parsing) */
    const char *mod_start = strstr(config_json, "\"module\"");
    if (!mod_start) {
        fprintf(stderr, "[python_bridge] config must contain \"module\" key\n");
        return RGPOT_PLUGIN_INVALID_PARAM;
    }
    /* Find the value string */
    mod_start = strchr(mod_start + 8, '"');
    if (!mod_start) return RGPOT_PLUGIN_INVALID_PARAM;
    mod_start++;
    const char *mod_end = strchr(mod_start, '"');
    if (!mod_end) return RGPOT_PLUGIN_INVALID_PARAM;

    char module_name[256];
    size_t len = (size_t)(mod_end - mod_start);
    if (len >= sizeof(module_name)) return RGPOT_PLUGIN_INVALID_PARAM;
    memcpy(module_name, mod_start, len);
    module_name[len] = '\0';

    PyPluginInstance *inst = calloc(1, sizeof(PyPluginInstance));
    if (!inst) return RGPOT_PLUGIN_ERROR;

    /* Import the Python module */
    inst->module = PyImport_ImportModule(module_name);
    if (!inst->module) {
        PyErr_Print();
        snprintf(inst->last_error, sizeof(inst->last_error),
                 "failed to import Python module '%s'", module_name);
        free(inst);
        return RGPOT_PLUGIN_ERROR;
    }

    /* Call plugin_create(config_json) -> int handle */
    PyObject *create_fn = PyObject_GetAttrString(inst->module, "plugin_create");
    if (!create_fn) {
        snprintf(inst->last_error, sizeof(inst->last_error),
                 "module '%s' missing plugin_create function", module_name);
        Py_DECREF(inst->module);
        free(inst);
        return RGPOT_PLUGIN_ERROR;
    }

    PyObject *args = Py_BuildValue("(s)", config_json);
    PyObject *result = PyObject_CallObject(create_fn, args);
    Py_DECREF(args);
    Py_DECREF(create_fn);

    if (!result || !PyLong_Check(result)) {
        PyErr_Print();
        Py_XDECREF(result);
        Py_DECREF(inst->module);
        free(inst);
        return RGPOT_PLUGIN_ERROR;
    }

    inst->py_handle = (int)PyLong_AsLong(result);
    Py_DECREF(result);

    *out = (rgpot_plugin_instance *)inst;
    return RGPOT_PLUGIN_OK;
}

static void py_destroy(rgpot_plugin_instance *raw) {
    PyPluginInstance *inst = (PyPluginInstance *)raw;
    if (!inst) return;

    if (inst->module) {
        PyObject *destroy_fn = PyObject_GetAttrString(inst->module, "plugin_destroy");
        if (destroy_fn) {
            PyObject *args = Py_BuildValue("(i)", inst->py_handle);
            PyObject *result = PyObject_CallObject(destroy_fn, args);
            Py_XDECREF(result);
            Py_DECREF(args);
            Py_DECREF(destroy_fn);
        }
        Py_DECREF(inst->module);
    }
    free(inst);
}

static rgpot_plugin_status_t
py_calculate(rgpot_plugin_instance *raw,
             const rgpot_plugin_input_t *input,
             rgpot_plugin_output_t *output) {
    PyPluginInstance *inst = (PyPluginInstance *)raw;
    if (!inst || !inst->module) return RGPOT_PLUGIN_ERROR;

    PyObject *calc_fn = PyObject_GetAttrString(inst->module, "plugin_calculate");
    if (!calc_fn) {
        snprintf(inst->last_error, sizeof(inst->last_error),
                 "module missing plugin_calculate function");
        return RGPOT_PLUGIN_ERROR;
    }

    /* Import numpy for array creation */
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np) { Py_DECREF(calc_fn); return RGPOT_PLUGIN_ERROR; }

    /* Create numpy arrays from input pointers */
    npy_intp pos_dims[1] = {(npy_intp)(input->n_atoms * 3)};
    npy_intp atm_dims[1] = {(npy_intp)(input->n_atoms)};
    npy_intp cell_dims[1] = {9};

    /* These are non-owning views -- numpy does not free the data */
    PyObject *pos_arr = PyArray_SimpleNewFromData(
        1, pos_dims, NPY_FLOAT64, (void *)input->positions);
    PyObject *atm_arr = PyArray_SimpleNewFromData(
        1, atm_dims, NPY_INT32, (void *)input->atomic_numbers);
    PyObject *cell_arr = PyArray_SimpleNewFromData(
        1, cell_dims, NPY_FLOAT64, (void *)input->cell);

    PyObject *args = Py_BuildValue("(iNNNN)",
        inst->py_handle, (int)input->n_atoms, pos_arr, atm_arr, cell_arr);

    /* Actually pass all 5 args: handle, n_atoms, positions, atomic_numbers, cell */
    args = PyTuple_New(5);
    PyTuple_SetItem(args, 0, PyLong_FromLong(inst->py_handle));
    PyTuple_SetItem(args, 1, PyLong_FromLong((long)input->n_atoms));
    PyTuple_SetItem(args, 2, pos_arr);
    PyTuple_SetItem(args, 3, atm_arr);
    PyTuple_SetItem(args, 4, cell_arr);

    PyObject *result = PyObject_CallObject(calc_fn, args);
    Py_DECREF(args);
    Py_DECREF(calc_fn);
    Py_DECREF(np);

    if (!result || !PyTuple_Check(result) || PyTuple_Size(result) != 2) {
        PyErr_Print();
        snprintf(inst->last_error, sizeof(inst->last_error),
                 "plugin_calculate must return (energy, forces_array)");
        Py_XDECREF(result);
        return RGPOT_PLUGIN_ERROR;
    }

    /* Extract energy (float) and forces (numpy array) */
    output->energy = PyFloat_AsDouble(PyTuple_GetItem(result, 0));

    PyObject *forces_arr = PyTuple_GetItem(result, 1);
    PyArrayObject *forces_np = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)forces_arr);

    double *fdata = (double *)PyArray_DATA(forces_np);
    memcpy(output->forces, fdata, input->n_atoms * 3 * sizeof(double));

    Py_DECREF(forces_np);
    Py_DECREF(result);

    output->variance = 0.0;
    return RGPOT_PLUGIN_OK;
}

static const char *py_last_error(const rgpot_plugin_instance *raw) {
    const PyPluginInstance *inst = (const PyPluginInstance *)raw;
    return inst ? inst->last_error : "";
}

static const rgpot_plugin_descriptor_t py_descriptor = {
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "python",
    "1.0.0",
    "angstrom",
    "eV",
    py_create,
    py_destroy,
    py_calculate,
    NULL, /* capabilities */
    py_last_error,
};

RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *
rgpot_plugin_init(void) {
    return &py_descriptor;
}
