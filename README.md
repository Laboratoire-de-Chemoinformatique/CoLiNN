## Combinatorial Library Neural Network (CoLiNN)

Welcome to the official repository for the Combinatorial Library Neural Network (CoLiNN) project!

### Overview
The visualization of combinatorial library chemical space is crucial in drug discovery, 
offering insights into available compound classes, their diversity, and physicochemical property distribution. 
Traditionally, this visualization requires extensive resources for compound enumeration, standardization, descriptor calculation, 
and dimensionality reduction.

In this study, we introduce CoLiNN, a neural network designed to predict the projection of compounds on a 2D chemical space map 
using only their building blocks and reaction information. This innovative approach eliminates the need for compound enumeration, streamlining the visualization process.

### Key Features
- **Efficient Visualization:** Predicts compound positions on 2D chemical space maps without the need for enumeration.
- **High Predictive Performance:** Trained on 2.5K virtual DNA-Encoded Libraries (DELs), CoLiNN accurately predicts compound positions on Generative Topographic Maps (GTMs).
- **Comparison with ChEMBL Database:** Demonstrates consistent similarity-based rankings between DELs and ChEMBL using both “true” and CoLiNN-predicted GTMs.
- **Enhanced Library Design:** Facilitates efficient exploration of library design space, making it a potential go-to tool for combinatorial compound library design.

### Applications
- **Drug Discovery:** Streamlines the identification of diverse and physicochemically appropriate compounds.
- **Combinatorial Library Design:** Allows for efficient exploration and comparison of different library designs without exhaustive enumeration.

### Getting Started
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/CoLiNN.git
   ```
2. **Install Dependencies:**
   ```bash
   conda env create -f environment.yml
   ```
3. **Run CoLiNN:**
   ```bash
   python main.py
   ```

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contact

For questions or suggestions, feel free to open an issue or contact us at
[reginapikalyova@gmail.com](mailto:reginapikalyova@gmail.com) or [tagirshin@gmail.com](mailto:tagirshin@gmail.com).

---

We hope CoLiNN will help streamline and enhance your combinatorial library design and visualization tasks!

