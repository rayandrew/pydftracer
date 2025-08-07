# pydftracer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A no operation (**No-Op**) Python binding for [DFTracer](https://github.com/hariharan-devarajan/dftracer) that provides seamless API compatibility without requiring the full DFTracer installation. Perfect for testing, development, and environments where you want to use DFTracer's API without the overhead of actual tracing.

## Purposes

- Are prototyping applications that will eventually use full DFTracer
- Maintain DFTracer API compatibility in environments where tracing is not needed (e.g. production apps)

## Installation

```bash
pip install dftracer
```

## Documentation

* Building DFTracer: [https://dftracer.readthedocs.io/en/latest/build.html](https://dftracer.readthedocs.io/en/latest/build.html)
* Integrating DFTracer: [https://dftracer.readthedocs.io/en/latest/examples.html](https://dftracer.readthedocs.io/en/latest/examples.html)
* Visualizing DFTracer Traces: [https://dftracer.readthedocs.io/en/latest/perfetto.html](https://dftracer.readthedocs.io/en/latest/perfetto.html)
* Building DFAnalyzer: [https://dftracer.readthedocs.io/en/latest/dfanalyzer_build.html](https://dftracer.readthedocs.io/en/latest/dfanalyzer_build.html)

## Citation and Reference

The original SC'24 paper describes the design and implementation of the DFTracer code. Please cite this paper and the code if you use DFTracer in your research. 

```
@inproceedings{devarajan_dftracer_2024,
    address = {Atlanta, GA},
    title = {{DFTracer}: {An} {Analysis}-{Friendly} {Data} {Flow} {Tracer} for {AI}-{Driven} {Workflows}},
    shorttitle = {{DFTracer}},
    urldate = {2024-07-31},
    booktitle = {{SC24}: {International} {Conference} for {High} {Performance} {Computing}, {Networking}, {Storage} and {Analysis}},
    publisher = {IEEE},
    author = {Devarajan, Hariharan and Pottier, Loic and Velusamy, Kaushik and Zheng, Huihuo and Yildirim, Izzet and Kogiou, Olga and Yu, Weikuan and Kougkas, Anthony and Sun, Xian-He and Yeom, Jae Seung and Mohror, Kathryn},
    month = nov,
    year = {2024},
}

@misc{devarajan_dftracer_code_2024,
    type = {Github},
    title = {Github {DFTracer}},
    shorttitle = {{DFTracer}},
    url = {https://github.com/LLNL/dftracer.git},
    urldate = {2024-07-31},
    journal = {DFTracer: A multi-level dataflow tracer for capture I/O calls from worklows.},
    author = {Devarajan, Hariharan and Pottier, Loic and Velusamy, Kaushik and Zheng, Huihuo and Yildirim, Izzet and Kogiou, Olga and Yu, Weikuan and Kougkas, Anthony and Sun, Xian-He and Yeom, Jae Seung and Mohror, Kathryn},
    month = jun,
    year = {2024},
}
```

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344; and under the auspices of the National Cancer Institute (NCI) by Frederick National Laboratory for Cancer Research (FNLCR) under Contract 75N91019D00024. This research used resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (DOE) Office of Science user facility at Argonne National Laboratory and is based on research supported by the U.S. DOE Office of Science-Advanced Scientific Computing Research Program, under Contract No. DE-AC02-06CH11357. Office of Advanced Scientific Computing Research under the DOE Early Career Research Program. Also, This material is based upon work partially supported by LLNL LDRD 23-ERD-045 and 24-SI-005. LLNL-CONF-857447.
