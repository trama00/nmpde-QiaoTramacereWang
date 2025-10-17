#pragma once

#include <iostream>
#include <memory>

namespace WaveEquation
{
    template <int dim>
    class WaveSolverBase
    {
    public:
        virtual ~WaveSolverBase() = default;

        // Main interface
        virtual void setup() = 0;
        virtual void run() = 0;
        
        // Analysis
        virtual double compute_energy() const = 0;
        
        // Output
        virtual void output_results(unsigned int step, double time) const = 0;

    protected:
        // Simple logging (no MPI for now)
        std::ostream &pcout = std::cout;
    };

} // namespace WaveEquation