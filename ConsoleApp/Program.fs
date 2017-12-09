module Program

open MnistDatabase
open MnistDatabaseExtraction
open MnistDatabaseNeuralNetwork
open System
open System.IO

let dataFilesTest = { Labels = @".\Data\t10k-labels-idx1-ubyte.gz"; Images = @".\Data\t10k-images-idx3-ubyte.gz" }
let dataFilesTrain = { Labels =  @".\Data\train-labels-idx1-ubyte.gz"; Images = @".\Data\train-images-idx3-ubyte.gz" }

let extractData () =
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

let trainForBestResult () =
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
    finalModel |> ignore
    printfn "Neural network execution completed."

let generateNumbersPareidolia () =
    printfn "Enter existing directory name to store image files:"
    let directory = Console.ReadLine()
    if not (Directory.Exists(directory)) then failwith "Directory must exist."
    let subfolder = Path.Combine(directory, "FSharpNeuralNetwork-Dreams")
    Directory.CreateDirectory(subfolder) |> ignore
    printfn "Short training completed."
    generateAndSaveNumbersPareidolia dataFilesTrain subfolder
    printfn "Images saved."
    ()

type CommandDeclaration = { key : int; title : string; execute : option<unit -> unit> }

[<EntryPoint>]
let main argv =
    argv |> ignore
    
    let declarations = [
        { key = 1; title = "Extract handwritten digits as image files."; execute = Some extractData }
        { key = 2; title = "Train neural network for best recognition result." ; execute = Some trainForBestResult }
        { key = 3; title = "Generate number Pareidolia images." ; execute = Some generateNumbersPareidolia }
        { key = 4; title = "Exit" ; execute = None }
    ]
    
    let rec inputIter () =
        printfn "This application demonstrates simple F# neural network that recognizes \"THE MNIST DATABASE of handwritten digits\" http://yann.lecun.com/exdb/mnist/."
        declarations |> List.iter (fun declaration -> printfn "%i - %s" declaration.key declaration.title)
        
        let commandString = Console.ReadLine()
        let commandOptionByString s = declarations |> List.tryFind (fun c -> c.key.ToString() = s) |> Option.map (fun x -> x.execute)
        match commandOptionByString commandString with
        | Some executeOption -> 
            match executeOption with 
            | Some execute ->
                execute ()
                inputIter ()
            | None -> ()
        | None -> 
            printfn "Unexpected command '%s'" commandString
    inputIter ()
    0 // return an integer exit code

