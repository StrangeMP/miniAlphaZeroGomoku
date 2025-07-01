# Mini AlphaZero Gomoku

Mini AlphaZero Gomoku is a lightweight implementation of the AlphaZero algorithm designed specifically for environments with constrained computational resources. This project focuses on applying Monte Carlo Tree Search (MCTS) with a quantized neural network to play the board game Gomoku. The implementation is written entirely in C++ with a custom quantized inference engine and efficient handling of neural network parameters at compile time.

## Overview

The project explores the balance between performance and computational efficiency by using a highly simplified neural network model. Key features include:

- **Quantized Neural Network Framework**: A compact and optimized neural network implementation in C++ that supports quantized operations, ensuring minimal computational overhead.
- **Neural Network Integration**: The neural network consists of two residual blocks with 32 filters per layer, trained externally and injected into the C++ implementation at compile time.
- **Monte Carlo Tree Search (MCTS)**: Efficient MCTS implementation designed to maximize the number of searches per second, essential for strong gameplay.
- **Modern C++ Techniques**: Extensively uses modern C++17 features such as smart pointers, constexpr, and template metaprogramming to ensure high performance and maintainability.
- **Model Export and Deployment**: Python scripts are used to parse and export trained model parameters into C++ header files, enabling seamless integration during compilation.
- **Focus on Practicality**: While the model surpasses average human performance, its simplicity limits its ability to recognize complex patterns, making it less competitive than specialized models like NNUE in certain scenarios.

## Features

- **Quantization**: Reduces the precision of neural network computations to significantly boost performance without compromising play quality.
- **Compile-Time Model Injection**: Embeds model parameters directly into the application, eliminating runtime model loading and improving efficiency.
- **Custom Inference Engine**: Implements core neural network operations from scratch, tailored specifically for Gomoku's board size and requirements.
- **Lightweight Design**: Optimized for environments with limited computational resources, making it ideal for educational or embedded systems.
- **Modern C++ Design**: Leverages advanced C++17 features for cleaner code, better safety, and optimized performance.

## Limitations

- The simplified model struggles with recognizing highly complex patterns due to its limited capacity (2 residual blocks, 32 filters per layer).
- Specialized models like NNUE may outperform this implementation in specific scenarios due to their tailored architectures.

## Lessons Learned

This project highlights the trade-offs between computational efficiency and model performance. While the quantized model achieves above-average human performance, its simplicity limits its ability to compete with specialized models. The experience underscores the importance of tailoring models to specific use cases to achieve optimal results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The implementation was inspired by the AlphaZero algorithm and adapted to fit the constraints of limited computational resources. Special thanks to the course instructors for guidance and feedback.
