# Simple README

## Compiling

To build the executable, make sure you have loaded the needed modules with

```bash
module load gcc-glibc dealii
```

Then run the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

The executable will be created into `build`, and can be executed through

```bash
./executable-name
```

## Result Verification

Here is a sample visualization of the results using Paraview:

- We set the parameters as follows:
  - `f = 0.0`
  - `g = 0.0`
  - `mu = 1.0`
  - `theta = 0.5`
  - `initial condition: u(x,0) = sin(pi*x)*sin(pi*y)*sin(pi*z); U'(x,0) = 0`
  
![Paraview](./src/OtherImplementation/Visual.png)
