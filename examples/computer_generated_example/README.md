# Example 2: Computer generated example

Directory containing code to reproduce example 2 from the paper.

```
computer_generated_example
└─── README.md
└─── simulation_example.ipynb
└─── colored_ee.jld
└─── table.jld
└─── white_ee.jld
└─── white_oe.jld
└─── timingsN.jld
└─── timingsNhidden.jld
```

## Description

-  ``simulation_example.ipynb``: Juyter notebook containing code for reproducing the example.
-  ``*.jld``: Data from previously executions. That is, some of the cells from the Jupyter notebook may take several hours to run. Hence we choose to store some results in order to avoid having to repeat the computation. If you want to repeat it, just delete the correspondent ``.jld`` file and, once the file is not found the path, the computation will be repeated.
