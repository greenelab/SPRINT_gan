# Privacy-preserving generative deep neural networks support clinical data sharing

Brett K. Beaulieu-Jones<sup>1</sup>, Zhiwei Steven Wu<sup>2</sup>, Chris Williams<sup>3</sup>, Casey S. Greene<sup>3</sup>*

<sup>1</sup>Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, Pennsylvania, USA.

<sup>2</sup>Computer and Information Science, School of Engineering and Applied Sciences, University of Pennsylvania, Philadelphia, Pennsylvania, USA.

<sup>3</sup>Department of Systems Pharmacology and Translational Therapeutics, Perelman School of Medicine, University of Pennsylvania, Philadelphia, Pennsylvania, USA.

To whom correspondence should be addressed: csgreene (at) upenn.edu

### https://doi.org/10.1101/159756

Introduction
--------
Though it is widely recognized that data sharing enables faster scientific progress, the sensible need to protect participant privacy hampers this practice in medicine. We train deep neural networks that generate synthetic subjects closely resembling study participants. Using the SPRINT trial as an example, we show that machine-learning models built from simulated participants generalize to the original dataset. We incorporate differential privacy, which offers strong guarantees on the likelihood that a subject could be identified as a member of the trial. Investigators who have compiled a dataset can use our method to provide a freely accessible public version that enables other scientists to perform discovery-oriented analyses. Generated data can be released alongside analytical code to enable fully reproducible workflows, even when privacy is a concern. By addressing data sharing challenges, deep neural networks can facilitate the rigorous and reproducible investigation of clinical datasets.

First run Preprocess.ipynb on the SPRINT clinical trial data (https://challenge.nejm.org/pages/home)

The results in the paper use:

`python dp_gan.py --noise 1 --clip_value 0.0001 --epochs 500 --lr 2e-05 --batch_size 1 --prefix 1_0.0001_2e-05_ `

Due to GPU threading, the current lack of an effective way to set a random seed for tensorflow, and the fact that we are optimizing two neural networks (unlikely to find global minima) it may require running multiple times to hit a good local minimum.

<img src="https://github.com/greenelab/SPRINT_gan/raw/master/figures/Figure_2A.png?raw=true" alt="Fig 2A" width="300"/><img src="https://github.com/greenelab/SPRINT_gan/raw/master/figures/Figure_2B.png?raw=true" alt="Fig 2B" width="300"/>
<img src="https://github.com/greenelab/SPRINT_gan/raw/master/figures/Figure_2C.png?raw=true" alt="Fig 2C" width="300"/><img src="https://github.com/greenelab/SPRINT_gan/raw/master/figures/Figure_2D.png?raw=true" alt="Fig 2D" width="300"/>

**Median Systolic Blood Pressure Trajectories from initial visit to 27 months.** **A.)** Simulated samples (private and non-private) generated from the final (500th) epoch of training. **B.)** Simulated samples generated from the epoch with the best performing logistic regression classifier. **C.)** Simulated samples from the epoch with the best performing random forest classifier. **D.)** Simulated samples from the top five random forest classifier epochs and top five logistic regression classifier epochs.


Feedback
--------

Please feel free to email us - (brettbe) at med.upenn.edu with any feedback or raise a github issue with any comments or questions.

Acknowledgements
----------------
Acknowledgments: We thank Jason H. Moore (University of Pennsylvania), Aaron Roth (University of Pennsylvania), Gregory Way (University of Pennsylvania), Yoseph Barash (University of Pennsylvania) and Anupama Jha (University of Pennsylvania) for their helpful discussions. We also thank the participants of the SPRINT trial and the entire SPRINT Research Group for providing the data used in this study. Funding: This work was supported by the Gordon and Betty Moore Foundation under a Data Driven Discovery Investigator Award to C.S.G. (GBMF 4552). B.K.B.-J. Was supported by a Commonwealth Universal Research Enhancement (CURE) Program grant from the Pennsylvania Department of Health and by US National Institutes of Health grants AI116794 and LM010098. Z.S.W is funded in part by a subcontract on the DARPA Brandeis project and a grant from the Sloan Foundation. 
