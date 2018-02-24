#define Py_LIMITED_API
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <vector>
#include <algorithm>
#include <iostream>

static PyObject *quantize_quantize(PyObject *self, PyObject *args);

static PyMethodDef QuantizeMethods[] = {
        {"quantize",  quantize_quantize, METH_VARARGS, "Quantize an array"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef quantizemodule = {
        PyModuleDef_HEAD_INIT,
        "quantize",
        "Module for quantizing",
        -1,
        QuantizeMethods
};

PyMODINIT_FUNC PyInit_quantize()
{
    auto module = PyModule_Create(&quantizemodule);
    import_array();
    return module;
}

template <class CType>
PyObject *templated_quantize(PyArrayObject *arr, int n){
    PyArrayObject *boundaries = NULL;
    PyArrayObject *quantized = NULL;
    auto dtype = PyArray_DTYPE(arr)->type_num;
    int nd = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_SHAPE(arr);
    npy_intp size = PyArray_SIZE(arr);	

    CType *dptr = (CType *) PyArray_DATA(arr);
    std::vector<CType> values;
    std::vector<CType> data;
    std::vector<CType> bounds;
    std::vector<CType> res;

    quantized = (PyArrayObject *) PyArray_SimpleNew(nd, dims, dtype);
    npy_intp bound_dims[1] = {n};
    boundaries = (PyArrayObject *) PyArray_SimpleNew(1, bound_dims, dtype);

    for (npy_intp i = 0; i < size; ++i) {
        data.push_back(dptr[i]);
        values.push_back(dptr[i]);
    }

    std::sort(values.begin(), values.end());

    for (int p = 0; p < n; ++p) {
        bounds.push_back(values[std::floor(size * p / n)]);
    }
    for (auto el : data) {
        int lo = 0;
        int hi = bounds.size();
        while (lo < hi) {
            int mid = std::floor((lo + hi) / 2);
            if (bounds[mid] < el) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        res.push_back(1. * lo);
    }

    for (npy_intp i = 0; i < bounds.size(); ++i) {
        npy_intp idx[1] = {i};
        *(CType *) PyArray_GetPtr(boundaries, idx) = bounds[i];
    }

    std::vector<npy_intp> idx(nd, 0);
    for (npy_intp i = 0; i < res.size(); ++i) {
        *(CType *) PyArray_GetPtr(quantized, idx.data()) = res[i];
        idx.back()++;
        for (auto k = nd - 1; k > 0; --k){
            if (idx[k] == dims[nd - k]){
                idx[k] = 0;
                idx[k - 1]++;
            }
        }
    }

    return Py_BuildValue("NN", boundaries, quantized);
}


static PyObject *quantize_quantize(PyObject *self, PyObject *args){
    PyObject *arg = NULL;
    PyArrayObject *arr = NULL;
    
    int n;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &arg, &n)) {
        PyErr_SetString(PyExc_TypeError, "Array and integer expected");
        return NULL;
    }

    arr = (PyArrayObject*)PyArray_FROM_OTF(arg, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);

    if (arr == NULL) {
        goto fail;
    } else {
        auto dtype = PyArray_DTYPE(arr)->type_num;

        if (dtype == NPY_DOUBLE) {
            auto result = templated_quantize<double>(arr, n);
            Py_DECREF(arr);
            return result;
        }
        if (dtype == NPY_FLOAT) {
            auto result = templated_quantize<float>(arr, n);
            Py_DECREF(arr);
            return result;
        }
        goto fail;
    }

    fail:
        Py_XDECREF(arr);
        PyErr_SetString(PyExc_TypeError, "Double or float expected");
        return NULL;
}
