# Domain adaptations pytorch lightning

```bash
python trainer.py --src mnist --tgt mnist_m --backbone lenet --model so --img_size 28 --lr 1e-2 --weight_decay 2.5e-5 --momentum 0.9 --optimizer sgd --batch_size 64 --epoch 20 --logger True --save True

python trainer.py --src svhn --tgt mnist --backbone svhn-dann --model so --img_size 32 --lr 1e-2 --weight_decay 2.5e-5 --momentum 0.9 --optimizer sgd --batch_size 128 --epoch 30 --logger True --save True

python trainer.py --src mnist --tgt mnist_m --backbone lenet --model dann --img_size 28 --lr 1e-2 --weight_decay 2.5e-5 --momentum 0.9 --optimizer sgd --batch_size 64 --epoch 30 --logger True --lr_schedule True --use_tgt_val True

python trainer.py --src svhn --tgt mnist --backbone svhn-dann --model dann --img_size 32 --lr 1e-2 --weight_decay 2.5e-5 --momentum 0.9 --optimizer sgd --batch_size 64 --epoch 30 --logger True --use_tgt_val True

python trainer.py --src mnist --tgt mnist_m --backbone lenet --model so --img_size 28 --lr 1e-3 --weight_decay 2.5e-5 --momentum 0.9 --optimizer adam --batch_size 64 --epoch 20 --logger True --save ./pretrained/mnist2mnist_m

python trainer.py --src mnist --tgt mnist_m --backbone lenet --model adda --load ./pretrained/mnist2mnist_m --img_size 28 --lr 1e-3 --weight_decay 2.5e-5 --momentum 0.9 --optimizer adam --batch_size 64 --epoch 20 --logger True --use_tgt_val True

python trainer.py --src svhn --tgt mnist --backbone svhn-adda --model so --img_size 32 --lr 1e-3 --weight_decay 2.5e-5 --momentum 0.9 --optimizer adam --batch_size 128 --epoch 20 --logger True --save ./pretrained/svhn2mnist

python trainer.py --src svhn --tgt mnist --backbone svhn-adda --model adda --load ./pretrained/svhn2mnist --img_size 32 --lr 1e-4 --weight_decay 2.5e-5 --momentum 0.9 --optimizer adam --batch_size 64 --epoch 20 --logger True --use_tgt_val True
```
