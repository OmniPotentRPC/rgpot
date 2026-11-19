`EnginePotential::load` no longer unloads the engine library when the load fails after the library opened, so a failed load leaves the library mapped like a successful one.
