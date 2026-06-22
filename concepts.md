# Mode-Selective Criticality: Terminology & Concepts

## 1. The Core Proposal
*   **Mode-Selective Criticality:** The central hypothesis of the research[cite: 1, 3]. It posits that in a deep neural network, broad patterns can survive to any depth (reaching criticality) while fine, granular details naturally die out[cite: 1, 3]. Crucially, the internal wiring of the network alone decides which patterns survive[cite: 1, 3].

## 2. The Physics-to-ML Dictionary
This section maps the physical concepts of phase transitions directly to the mechanics of a deep neural network[cite: 1, 3].

*   **Distance / Time $\rightarrow$ Network Depth:** 
    *   *Physics:* The physical space between particles or the evolution of a system over time[cite: 1, 3].
    *   *ML Equivalent:* The number of hidden layers a signal passes through from the input layer to the output layer[cite: 1, 3].
*   **Two-Point Correlator $\rightarrow$ Activation Covariance:**
    *   *Physics:* A correlation function, $g(r)$, that measures how the state of a field at one point relates to a point separated by distance $r$[cite: 1, 3].
    *   *ML Equivalent:* Denoted as $Q^l(\Delta)$, it measures how the firing (activation) of one neuron relates to another neuron located $\Delta$ steps away within the exact same layer $l$[cite: 1, 3].
*   **Structure Factor $\rightarrow$ Fourier Transform:**
    *   *Physics:* The Fourier transform of the physical correlator, denoted as $S(k)$, which helps identify repeating structures[cite: 1, 3].
    *   *ML Equivalent:* The mathematical transformation of the network's activations into frequency space, denoted as $\tilde{Q}^l(k)$[cite: 1, 3].
*   **Correlation Length $\rightarrow$ Survival Depth:**
    *   *Physics:* The maximum distance ($\xi$) over which particles influence each other[cite: 1, 3]. As it approaches a critical point, this length diverges to infinity[cite: 1, 3].
    *   *ML Equivalent:* Denoted as $\xi_c$, this is the number of layers a specific data pattern can survive before washing out[cite: 1, 3]. 
*   **Critical Point $\rightarrow$ Edge of Chaos:**
    *   *Physics:* A phase transition state where a system's correlation length diverges, allowing influence to reach arbitrarily far[cite: 1, 3].
    *   *ML Equivalent:* A precisely balanced network state where a pattern's survival depth becomes infinite, allowing the signal to propagate through endless layers without fading or scrambling[cite: 1, 3].

## 3. Network States, Structure & Measurement
*   **Normal Modes $\rightarrow$ Fourier Modes Kept:**
    *   Just as physical translation symmetry decouples waves into independent "normal modes," the structural symmetry of a neural network decouples data into independent patterns or "Fourier modes"[cite: 1, 3]. Each mode gets its own distinct critical point[cite: 1, 3].
*   **Crystal $\rightarrow$ Symmetric Ring:**
    *   A perfectly symmetrical network architecture[cite: 1, 3]. Like a physical crystal, the modes it keeps are distinct, mathematically clean, and act independently[cite: 1, 3].
*   **Liquid $\rightarrow$ Broken Symmetry:**
    *   A disordered, messy network architecture (more like real-world models)[cite: 1, 3]. When symmetry is broken, the independent modes blur and smear together, behaving like a liquid[cite: 1, 3].
*   **Scattering Experiment $\rightarrow$ Sparse Autoencoder:**
    *   *Physics:* An experimental technique used to measure and untangle complex molecular structures in materials[cite: 1, 3].
    *   *ML Equivalent:* An AI instrument used to "read off" or recover the hidden, blurred Fourier modes from a disordered, liquid-like neural network where the clean mathematical theory breaks down[cite: 1, 3].