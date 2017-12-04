module Program

open MnistDatabase
let data_Test10kImages = @".\Data\t10k-images-idx3-ubyte.gz"
let data_Test10kLabels = @".\Data\t10k-labels-idx1-ubyte.gz"
let data_TrainImages = @".\Data\train-images-idx3-ubyte.gz"
let data_TrainLabels = @".\Data\train-labels-idx1-ubyte.gz"

[<EntryPoint>]
let main argv =
    // Write all test images
    MnistDatabaseExtraction.mnistLabeledImages 
        data_Test10kImages
        data_Test10kLabels
        @"C:\Temp\Images"

    // let a = readMnistDatabaseFromCompressedFile fileName |> Array.ofSeq
    // printfn "%A" a.Length
    
    //let a = AppDomain.CurrentDomain.GetAssemblies()
    //let b = a
    0 // return an integer exit code

