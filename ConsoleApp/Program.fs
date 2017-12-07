module Program

open MnistDatabase
open MnistDatabaseExtraction
open MnistDatabaseNeuralNetwork
open System
open System.IO

let dataFilesTest = { Labels = @".\Data\t10k-labels-idx1-ubyte.gz"; Images = @".\Data\t10k-images-idx3-ubyte.gz" }
let dataFilesTrain = { Labels =  @".\Data\train-labels-idx1-ubyte.gz"; Images = @".\Data\train-images-idx3-ubyte.gz" }

[<EntryPoint>]
let main argv =
    argv |> ignore
    let extreactionDataCommand = "1"
    let trainingAndQueringTest = "2"
    printfn "This application demonstrates simple F# neural network that recognizes \"THE MNIST DATABASE of handwritten digits\" http://yann.lecun.com/exdb/mnist/."
    printfn "Press %s to extract handwritten digits as image files." extreactionDataCommand
    printfn "Press %s to run neural network training and recognition." trainingAndQueringTest
    let command = Console.ReadLine()
    if command = extreactionDataCommand then
        printfn "Enter existing directory name to store image files:"
        let directory = Console.ReadLine()
        if not (Directory.Exists(directory)) then failwith "Directory must exist."
        let subfolder = Path.Combine(directory, "FSharpNeuralNetwork-MNIST-dataset-images")
        Directory.CreateDirectory(subfolder) |> ignore
        let extractToNewFolder dataFiles directoryToCreate =
            Directory.CreateDirectory(directoryToCreate) |> ignore
            extractMnistDatabaseLabeledImages dataFiles directoryToCreate
        extractToNewFolder dataFilesTest (Path.Combine(subfolder, "Test"))
        extractToNewFolder dataFilesTrain (Path.Combine(subfolder, "Train"))
        printfn "Files extraction completed."
    else if command = trainingAndQueringTest then
        // Update progress only when next one percent or total records is processed.
        let logPercentProgress (percentUpdated : int -> unit) =
            let logTrainingProgress recordCount = 
                let mutable percentOriginal = -1;
                let logIter currentIndex = 
                    let percentCurrent = (currentIndex |> float) / (recordCount |> float) * 100.0 |> int
                    if percentCurrent <> percentOriginal then
                        percentOriginal <- percentCurrent
                        percentUpdated percentCurrent
                    ()
                logIter
            logTrainingProgress
        let testingLearningEpoch epoch model =     
            if epoch > 0 then // Skip initial model    
                printfn "Testing model from %i epoch:" epoch
                test model dataFilesTest |> (fun r -> r * 100.0 |> printfn "Learning performance score: %.2f%%")
        let finalModel = trainDefaultEpochs dataFilesTrain (logPercentProgress (printfn "Training progress: %i%%")) testingLearningEpoch
        finalModel |> ignore // We already have tested all epoches.
        printfn "Neural network execution completed."
    else
        printfn "Unexpected command '%s'" command
    Console.ReadKey() |> ignore
    printfn "Press any key to exit."
    0 // return an integer exit code

