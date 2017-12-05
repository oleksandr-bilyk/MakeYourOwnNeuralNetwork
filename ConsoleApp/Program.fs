module Program

open MnistDatabase
open MnistDatabaseExtraction

let dataFilesTest = { Labels = @".\Data\t10k-labels-idx1-ubyte.gz"; Images = @".\Data\t10k-images-idx3-ubyte.gz" }
let dataFilesTrain = { Labels =  @".\Data\train-labels-idx1-ubyte.gz"; Images = @".\Data\train-images-idx3-ubyte.gz" }

[<EntryPoint>]
let main argv =
    argv |> ignore
    // Write all test images
    extractMnistDatabaseLabeledImages dataFilesTest @"C:\Temp\Images\Test"
    extractMnistDatabaseLabeledImages dataFilesTrain @"C:\Temp\Images\Train"

    // let a = readMnistDatabaseFromCompressedFile fileName |> Array.ofSeq
    // printfn "%A" a.Length
    
    //let a = AppDomain.CurrentDomain.GetAssemblies()
    //let b = a
    0 // return an integer exit code

