# dMA-NAS-UTSF
  Although deep learning has made remarkable progress in time series forecasting, enormous hyperparameters consume a lot of effort to tune. Moreover, to further build the forecasting models with better performance, time series decomposition is usually adopted to mine implicit patterns of the data. Inspired by the time series decomposition, automatically searching for a network architecture after decomposing the time series is proposed. The searching process is non-trivial and has two key challenges: 1) impairment of time series information after decomposing and 2) enlarged search space caused by the huge parameters to be optimized. In this paper, a decomposition-based memetic neural architecture search algorithm is proposed for univariate time series forecasting to address these two challenges. For the first challenge, a general univariate time series forecasting paradigm is designed as the building pipeline of the individual in the proposed algorithm, which considers both the decomposed components and the original series as the compensation information to improve the network representation ability. For the second challenge, with the intrinsic property of representation of individuals in mind, we design a decomposition-based memetic algorithm with a discriminative local search operator to automatically optimize the network configurations. The experimental results on nine benchmarks with four horizons and one application of remaining useful forecasting demonstrate that the discovered architectures by the proposed algorithm achieve competitive performance compared with six methods under aligned settings.

![图片](https://github.com/user-attachments/assets/cce86af7-fffa-4be8-8d7a-552f50e27428)

  

## Highlights
(1) This paper is the first attempt to perform neural architecture search on the decomposed time series, which not only uses the characteristics of the decomposed time series to achieve high-precision forecasting results, but also automatically searches for the optimal network settings, reducing the effort to tune a large number of parameters; 

(2) This paper achieves a substantial improvement on the search space of neural architecture search techniques for time series forecasting. Existing methods only search for partial parameters under a fixed network, while this paper simultaneously optimizes network types of each decomposition series and their internal parameters, greatly improving the forecasting performance. 

(3) The experimental results demonstrate that the architectures searched by the proposed algorithm achieve amazing performance compared with state-of-the-arts under aligned settings. Moreover, the automated search process of the proposed algorithm relieves experts from the tedious process of trial and error of network parameters, saving a lot of effort.

## Implementation Steps
Step 1. Download the corresponding dataset to the datasets folder.

Step 2. Enter the corresponding parameters in main.py.

Step 3. Run main.py for code execution.

