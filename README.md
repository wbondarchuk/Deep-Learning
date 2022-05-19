# Convolutional neural networks for artistic style transfer




## Setup

1. Install Python (2.7), pip and virtualenv on your machine. The
instructions to do this depend on your operating system (Linux, macOS,
Windows), but there are many tutorials on the internet that should
help you get started.

2. Once you have the above setup, it is quite easy to setup the
requirements for the notebooks in this repository. First you clone a
copy of this repository:

   ````
   git clone https://github.com/hnarayanan/artistic-style-transfer.git
   ````

3. Then you navigate to this folder in your shell and then install the
requirements needed for the Jupyter notebooks.

   ````
   cd artistic-style-transfer
   virtualenv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ````

4. If it doesn't exist, create a file called `~/.keras/keras.json` and
make sure it looks like the following:

   ````
   {
       "image_dim_ordering": "tf",
       "epsilon": 1e-07,
       "floatx": "float32",
       "backend": "tensorflow"
   }
   ````

5. That's it! You can now start Jupyter and browse, open, run and
modify the notebooks.

   ````
   jupyter notebook
   ````


[blog-post]: https://harishnarayanan.org/writing/artistic-style-transfer/
[talk-slides]: https://speakerdeck.com/hnarayanan/convolutional-neural-networks-for-artistic-style-transfer

