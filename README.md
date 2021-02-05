# Domain Adaptation Implementations
Domain Adversarial Training of Neural Network (DANN) [paper](https://arxiv.org/pdf/1505.07818.pdf)

# Implementation References
[https://github.com/CuthbertCai/pytorch_DANN](https://github.com/CuthbertCai/pytorch_DANN)

# Experiments Reproduction

```bash
# MNIST to MNIST-M
python experiment.py --src mnist --tgt mnist_m --model dann --lr 1e-2 --epoch 20 --batch_size 64

# SVHN to MNIST
python experiment.py --src svhn --tgt mnist --model dann --lr 2e-2 --epoch 100 --batch_size 32 --lr_scheduler False
```
