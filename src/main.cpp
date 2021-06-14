#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "inference.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;


PYBIND11_MODULE(ort_inference, m) {
    py::class_<inference::SequenceClassificationOrtInference>(m, "SequenceClassificationOrtInference")
        .def(py::init<std::string, int>())
        .def("batch_predict", &inference::SequenceClassificationOrtInference::batch_predict)
        .def("predict", &inference::SequenceClassificationOrtInference::predict);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
