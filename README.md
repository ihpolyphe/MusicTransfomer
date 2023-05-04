# MusicTransfomer Setup
# Environment
OS: Windows11  
WSL2: 5.10.16.3-microsoft-standard-WSL2(Ubuntu 20.04)  
CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz  

## Install magenta On WSL2
```
$ curl https://raw.githubusercontent.com/tensorflow/magenta/main/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
$ bash /tmp/magenta-install.sh
```
reopen terminal and use command with
```
$ source activate magenta
```
clone magenta
```
$ git clone https://github.com/magenta/magenta.git
$ cd magenta 
$ pip install -e .
```

## Downgrrade magenta
```
$ pip install tensorflow==1.15.2
$ pip install magenta==1.3.1
$ pip install tensor2tensor==1.15.5
$ pip install tensorflow-probability==0.7.0
$ pip install tensorflow-datasets==3.0.0
$ pip install pip install numpy==1.19.5
```
