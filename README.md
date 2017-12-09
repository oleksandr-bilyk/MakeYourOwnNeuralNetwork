# Make Your Own Neural Network With F# programming language
F# Code developed using as refactoring of Python code from book ["Make Your Own Neural Network"](https://www.goodreads.com/book/show/29746976-make-your-own-neural-network)

Repository contains original binary data of "THE MNIST DATABASE of handwritten digits" (http://yann.lecun.com/exdb/mnist/)
To run application you should be familiar with:
- [Visual Studio Code](https://code.visualstudio.com/)
- Visual Studio Code [Ionide extension](https://docs.microsoft.com/en-us/dotnet/fsharp/get-started/get-started-vscode) for F#
- [.NET Core 2.0](https://code.visualstudio.com/docs/other/dotnet)
- [Paket](https://fsprojects.github.io/Paket/getting-started.html)

Console Application (ConsoleApp folder) has programe with two options:
1. Extract handwritten digits as image files. There are 60.000 images for training and 10.000 images for testing.

<![Extracted files](https://github.com/oleksandr-bilyk/MakeYourOwnNeuralNetwork/blob/master/doc/ExtractedFiles.png)>

2. Run neural network training and recognition. It takes few minutes to train neural network and testing functionality shows Learning performance score above 97% of right recognitions.
3. Generate number [Pareidolia](https://en.wikipedia.org/wiki/Pareidolia) images for numbers 0-9 range.

<![Extracted files](https://github.com/oleksandr-bilyk/MakeYourOwnNeuralNetwork/blob/master/doc/Pareidolia.png)>

Hope you will anjoy it :) 
Regards, Oleksandr Bilyk
