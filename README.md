# Phonon RIXS using Green's functions
This project uses the Momentum Average approximation to calculate the RIXS (Resonant Inelastic X-ray Scattering) cross-section for phonon models using lattice Green's functions. The following modeules are included:

### holstein
This is the basic model for a single itinerant electron creating an arbitarily large cloud in a highly constrained geometry around the core-hole site. This is based on the logic of polaronic self-trapping where the existing phonons bind the electron and prevent it from escaping and spreading the cloud in a larger area. The phonons are localised (non-dispersive) Einstein modes and the electron-phonon coupling is also momentum independent. This simple toy model can be solved using continued fractions and this solution is implemented here. For details see: [SciPost Phys. **11**, 062 (2021)](https://arxiv.org/abs/2011.05400).

### optical
This is a slightly more complicated model, where the phonons are assumed to be optical, i.e., have a tiny dispersion around the Einstein mode. Using some tricks, this can be generalised from the Holstein case above. For details see: [Phys. Rev. B **105**, L180302 (2022)](https://arxiv.org/abs/2201.04577).

### lang_firsov
This is an exact solution by cannonical transformation known as the Lang-Firsov theory. Only applicable for a single site approximation of the Holstein model, it has the obvious advantage of simplicity.
